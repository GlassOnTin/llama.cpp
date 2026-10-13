// llama-share-prewarm: a standalone program (links ggml only, no llama) that seeds the
// llama-share-registry's shared segments from a manifest + a GGUF file, before any
// llama.cpp process has loaded the model.
//
// Every manifest-matched weight tensor is allocated as its own CUDA VMM segment on the
// shareable buffer type ggml_backend_cuda_shareable_buffer_type(), filled by pread from the GGUF data
// section, exported, and registered under (digest, bus id). A llama.cpp process started
// afterwards with --share-ipc attaches instead of publishing: even its first load is the
// warm path, with no hashing and no private upload.
//
// Digests come from the manifest, so the preloader never hashes; correctness rests on
// the manifest having been written from a publish run of the same weight bytes (the same
// trust boundary as the loader's own attach path). Tensors the registry can already
// supply are skipped, so prewarming a pool that is still resident is a fast no-op.
//
// usage: llama-share-prewarm <socket> <manifest.tsv> <model.gguf>
//
// needs CUDA_VISIBLE_DEVICES set to one GPU (segment bus ids are keyed per device).

#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"

// the registry's wire format lives in the llama.cpp tree; the header keeps the protocol
// in sync by construction (contains no other llama dependency)
#include "llama-share.h"

#include <array>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

#include <fcntl.h>
#include <inttypes.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>

namespace {

// ---------------------------------------------------------------------------
// registry wire exchange, byte-for-byte the llama-share.cpp protocol
// (one message per connection)
// ---------------------------------------------------------------------------

int unix_connect(const std::string & path) {
    sockaddr_un addr = {};
    addr.sun_family = AF_UNIX;
    if (path.size() >= sizeof(addr.sun_path)) {
        return -1;
    }
    strncpy(addr.sun_path, path.c_str(), sizeof(addr.sun_path) - 1);

    const int fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) {
        return -1;
    }
    if (connect(fd, (const sockaddr *) &addr, sizeof(addr)) != 0) {
        close(fd);
        return -1;
    }
    return fd;
}

// send one frame: hdr + trailing payload, optionally carrying fd via SCM_RIGHTS
bool send_frame(int fd, const llama_share_hdr & hdr, const std::string & trailing,
                bool have_fd, int fd_out) {
    std::vector<uint8_t> frame(sizeof(hdr) + trailing.size());
    memcpy(frame.data(), &hdr, sizeof(hdr));
    memcpy(frame.data() + sizeof(hdr), trailing.data(), trailing.size());

    iovec iov = { frame.data(), frame.size() };
    union {
        cmsghdr h;
        char buf[CMSG_SPACE(sizeof(int))];
    } cmsg_buf = {};
    msghdr msg = {};
    msg.msg_iov = &iov;
    msg.msg_iovlen = 1;
    if (have_fd) {
        msg.msg_control = cmsg_buf.buf;
        msg.msg_controllen = sizeof(cmsg_buf);
        cmsghdr * cmsg = CMSG_FIRSTHDR(&msg);
        if (cmsg == nullptr) {
            return false;
        }
        cmsg->cmsg_level = SOL_SOCKET;
        cmsg->cmsg_type  = SCM_RIGHTS;
        cmsg->cmsg_len   = CMSG_LEN(sizeof(int));
        memcpy(CMSG_DATA(cmsg), &fd_out, sizeof(int));
    }
    while (true) {
        const ssize_t n = sendmsg(fd, &msg, MSG_NOSIGNAL);
        if (n < 0 && errno == EINTR) {
            continue;
        }
        return n == (ssize_t) frame.size();
    }
}

// receive one frame: hdr + trailing (name bytes, then bus bytes), optionally an fd
bool recv_frame(int fd, llama_share_hdr & hdr, std::string & name, std::string & bus, int * fd_in) {
    *fd_in = -1;

    const size_t payload_max = 4096; // name_len + bus_len sanity cap
    std::vector<uint8_t> frame(sizeof(hdr) + payload_max);
    size_t got = 0;
    bool have_hdr = false;

    while (true) {
        size_t want = sizeof(hdr);
        if (have_hdr) {
            const size_t payload_len = hdr.name_len + hdr.bus_len;
            if (payload_len > payload_max) {
                return false;
            }
            want = sizeof(hdr) + payload_len;
        }
        if (got >= want) {
            break;
        }

        iovec iov = { frame.data() + got, want - got };
        union {
            cmsghdr h;
            char buf[CMSG_SPACE(sizeof(int))];
        } cmsg_buf = {};
        msghdr msg = {};
        msg.msg_iov = &iov;
        msg.msg_iovlen = 1;
        msg.msg_control = cmsg_buf.buf;
        msg.msg_controllen = sizeof(cmsg_buf);

        const ssize_t n = recvmsg(fd, &msg, 0);
        if (n < 0 && errno == EINTR) {
            continue;
        }
        if (n <= 0) {
            return false;
        }
        got += n;
        for (cmsghdr * cmsg = CMSG_FIRSTHDR(&msg); cmsg != nullptr; cmsg = CMSG_NXTHDR(&msg, cmsg)) {
            if (cmsg->cmsg_level == SOL_SOCKET && cmsg->cmsg_type == SCM_RIGHTS &&
                    cmsg->cmsg_len >= CMSG_LEN(sizeof(int))) {
                int recvd = -1;
                memcpy(&recvd, CMSG_DATA(cmsg), sizeof(int));
                if (*fd_in < 0) {
                    *fd_in = recvd;
                } else {
                    close(recvd); // never expected; don't leak
                }
            }
        }
        if (!have_hdr && got >= sizeof(hdr)) {
            memcpy(&hdr, frame.data(), sizeof(hdr));
            have_hdr = true;
        }
    }

    std::string trailing((const char *) frame.data() + sizeof(hdr), hdr.name_len + hdr.bus_len);
    name.assign(trailing, 0, hdr.name_len);
    bus.assign(trailing, hdr.name_len, hdr.bus_len);
    return true;
}

// full round trip: connect, send request (optionally with one fd), read the reply
bool registry_roundtrip(const std::string & ipc_path, llama_share_hdr & req, const std::string & trailing,
                        int req_fd, llama_share_hdr & rep, std::string & rep_name,
                        std::string & rep_bus, int * rep_fd) {
    const int sfd = unix_connect(ipc_path);
    if (sfd < 0) {
        return false;
    }
    if (!send_frame(sfd, req, trailing, req_fd >= 0, req_fd)) {
        close(sfd);
        return false;
    }
    const bool ok = recv_frame(sfd, rep, rep_name, rep_bus, rep_fd) &&
                    rep.magic == LLAMA_SHARE_MAGIC && rep.version == LLAMA_SHARE_VERSION;
    close(sfd);
    return ok;
}

bool registry_query(const std::string & ipc_path, const uint8_t * digest, const std::string & bus) {
    llama_share_hdr req = {};
    req.magic   = LLAMA_SHARE_MAGIC;
    req.version = LLAMA_SHARE_VERSION;
    req.type    = LLAMA_SHARE_QUERY;
    req.bus_len = (uint32_t) bus.size() + 1;
    memcpy(req.digest, digest, 32);

    llama_share_hdr rep;
    std::string rep_name, rep_bus;
    int fd = -1;
    if (!registry_roundtrip(ipc_path, req, bus + std::string(1, '\0'), -1, rep, rep_name, rep_bus, &fd)) {
        return false;
    }
    const bool hit = rep.type == LLAMA_SHARE_QUERY && rep.status == LLAMA_SHARE_STATUS_HIT && fd >= 0;
    if (fd >= 0) {
        close(fd); // prewarm never attaches; consuming the fd closes it
    }
    return hit;
}

bool registry_register(const std::string & ipc_path, const uint8_t * digest, const std::string & bus,
                       uint64_t seg_size, const char * name, int fd) {
    llama_share_hdr req = {};
    req.magic    = LLAMA_SHARE_MAGIC;
    req.version  = LLAMA_SHARE_VERSION;
    req.type     = LLAMA_SHARE_REGISTER;
    req.name_len = (uint32_t) strlen(name) + 1;
    req.bus_len  = (uint32_t) bus.size() + 1;
    req.seg_size = seg_size;
    memcpy(req.digest, digest, 32);

    std::string trailing(name, strlen(name) + 1);
    trailing += bus + std::string(1, '\0');

    llama_share_hdr rep;
    std::string rep_name, rep_bus;
    int rep_fd = -1;
    if (!registry_roundtrip(ipc_path, req, trailing, fd, rep, rep_name, rep_bus, &rep_fd) ||
            rep.type != LLAMA_SHARE_REGISTER || rep.status != LLAMA_SHARE_STATUS_OK) {
        // the registry may or may not have taken the fd; closing ours is always safe
        close(fd);
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// manifest (name -> digest hex -> nbytes), same format llama-share.cpp writes
// ---------------------------------------------------------------------------

struct manifest_entry {
    std::string digest_hex; // 64 hex chars
    size_t      nbytes;
};

bool parse_manifest(const std::string & path, std::unordered_map<std::string, manifest_entry> & out) {
    FILE * f = fopen(path.c_str(), "r");
    if (f == nullptr) {
        return false;
    }
    char buf[4096];
    bool first = true;
    while (fgets(buf, sizeof(buf), f)) {
        if (first && strncmp(buf, "#gguf-share-manifest", 20) != 0) {
            fprintf(stderr, "prewarm: %s is not a share manifest\n", path.c_str());
            fclose(f);
            return false;
        }
        first = false;
        std::string line(buf);
        while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) {
            line.pop_back();
        }
        if (line.empty() || line[0] == '#') {
            continue;
        }
        const size_t t1 = line.find('\t');
        const size_t t2 = t1 == std::string::npos ? std::string::npos : line.find('\t', t1 + 1);
        if (t1 == std::string::npos || t2 == std::string::npos) {
            fprintf(stderr, "prewarm: malformed manifest line '%s'\n", line.c_str());
            fclose(f);
            return false;
        }
        manifest_entry e;
        e.digest_hex = line.substr(t1 + 1, 64);
        e.nbytes     = (size_t) strtoull(line.c_str() + t2 + 1, nullptr, 10);
        if (e.digest_hex.size() != 64 || e.nbytes == 0 ||
                line.find('\t', t2 + 1) != std::string::npos) {
            fprintf(stderr, "prewarm: malformed manifest entry for '%s'\n", line.substr(0, t1).c_str());
            fclose(f);
            return false;
        }
        bool hex_ok = true;
        for (char c : e.digest_hex) {
            if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) {
                hex_ok = false;
            }
        }
        if (!hex_ok) {
            fprintf(stderr, "prewarm: bad digest for '%s'\n", line.substr(0, t1).c_str());
            fclose(f);
            return false;
        }
        out[line.substr(0, t1)] = std::move(e);
    }
    fclose(f);
    return !out.empty();
}

void digest_from_hex(const std::string & hex, uint8_t * out) {
    auto nib = [](char c) {
        return c <= '9' ? c - '0' : c - 'a' + 10;
    };
    for (int i = 0; i < 32; i++) {
        out[i] = (uint8_t) ((nib(hex[2*i]) << 4) | nib(hex[2*i + 1]));
    }
}

double now_s() {
    timespec ts = {};
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

} // namespace

// ---------------------------------------------------------------------------

int main(int argc, char ** argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s <socket> <manifest.tsv> <model.gguf>\n", argv[0]);
        return 1;
    }
    const std::string socket_path   = argv[1];
    const std::string manifest_path = argv[2];
    const std::string gguf_path     = argv[3];

    std::unordered_map<std::string, manifest_entry> manifest;
    if (!parse_manifest(manifest_path, manifest)) {
        fprintf(stderr, "prewarm: cannot parse manifest %s\n", manifest_path.c_str());
        return 1;
    }
    printf("prewarm: manifest %s: %zu entries\n", manifest_path.c_str(), manifest.size());

    // CUDA backend + shareable buffer type, proc-addressed like the loader does
    ggml_backend_load_all();
    ggml_backend_reg_t reg = ggml_backend_reg_by_name("CUDA");
    if (reg == nullptr) {
        fprintf(stderr, "prewarm: no CUDA backend registered\n");
        return 1;
    }

    using bus_id_fn_t         = const char * (*)(int, char *, size_t);
    using shareable_buft_fn_t = ggml_backend_buffer_type_t (*)(int);
    using export_fn_t         = int (*)(ggml_backend_buffer_t, int *, size_t *);

    auto * fn_bus    = (bus_id_fn_t)         ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_shareable_device_bus_id");
    auto * fn_buft   = (shareable_buft_fn_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_shareable_buffer_type");
    auto * fn_export = (export_fn_t)         ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_buffer_export_shareable");
    if (fn_bus == nullptr || fn_buft == nullptr || fn_export == nullptr) {
        fprintf(stderr, "prewarm: CUDA backend lacks the shareable-buffer API\n");
        return 1;
    }

    const int device = 0; // CUDA_VISIBLE_DEVICES maps the target GPU to index 0
    char bus[64] = {};
    if (fn_bus(device, bus, sizeof(bus)) == nullptr) {
        fprintf(stderr, "prewarm: device %d has no VMM support\n", device);
        return 1;
    }
    ggml_backend_buffer_type_t buft = fn_buft(device);
    if (buft == nullptr) {
        fprintf(stderr, "prewarm: no shareable buffer type for device %d\n", device);
        return 1;
    }
    printf("prewarm: device %d, bus %s, shareable buffer type %s\n", device, bus, ggml_backend_buft_name(buft));

    // sanity check the registry (also fails fast when it is absent)
    {
        uint8_t zero[32] = {};
        llama_share_hdr req = {};
        req.magic   = LLAMA_SHARE_MAGIC;
        req.version = LLAMA_SHARE_VERSION;
        req.type    = LLAMA_SHARE_QUERY;
        req.bus_len = (uint32_t) strlen(bus) + 1;
        memcpy(req.digest, zero, sizeof(zero));
        llama_share_hdr rep;
        std::string rep_name, rep_bus;
        int rep_fd = -1;
        if (!registry_roundtrip(socket_path, req, bus + std::string(1, '\0'), -1, rep, rep_name, rep_bus, &rep_fd)) {
            fprintf(stderr, "prewarm: cannot reach registry at %s\n", socket_path.c_str());
            return 1;
        }
        if (rep_fd >= 0) {
            close(rep_fd);
        }
    }

    gguf_init_params ip = { /*no_alloc*/ true, /*ctx*/ nullptr };
    gguf_context * gf = gguf_init_from_file(gguf_path.c_str(), ip);
    if (gf == nullptr) {
        fprintf(stderr, "prewarm: cannot read %s\n", gguf_path.c_str());
        return 1;
    }
    const int64_t n_tensors = gguf_get_n_tensors(gf);
    const size_t  data_off  = gguf_get_data_offset(gf);

    struct work {
        int64_t                 gguf_idx;
        std::string             name;
        std::array<uint8_t, 32> digest;
        size_t                  nbytes;
    };
    std::vector<work> todo;
    uint64_t missing_names = 0, matched = 0, resident = 0, size_mismatched = 0;

    // pass A: classify the gguf's tensors against the manifest and a warm registry
    for (int64_t i = 0; i < n_tensors; i++) {
        const char * name = gguf_get_tensor_name(gf, i);
        auto it = manifest.find(name);
        if (it == manifest.end()) {
            if (missing_names < 5) {
                printf("prewarm: no manifest entry for '%s' (%zu bytes), it will not be shared\n",
                       name, gguf_get_tensor_size(gf, i));
            }
            missing_names++;
            continue;
        }
        if (it->second.nbytes != gguf_get_tensor_size(gf, i)) {
            size_mismatched++;
            continue;
        }
        matched++;
        work w;
        w.gguf_idx = i;
        w.name     = name;
        w.nbytes   = it->second.nbytes;
        digest_from_hex(it->second.digest_hex, w.digest.data());
        if (registry_query(socket_path, w.digest.data(), bus)) {
            resident++;
            continue;
        }
        todo.push_back(std::move(w));
    }
    printf("prewarm: %" PRId64 " gguf tensors; %" PRIu64 " matched (already resident %"
           PRIu64 ", to seed %zu), %"
           PRIu64 " without manifest entries, %" PRIu64 " size-mismatched\n",
           n_tensors, matched, resident, todo.size(), missing_names, size_mismatched);

    if (todo.empty()) {
        printf("prewarm: nothing to seed\n");
        gguf_free(gf);
        return 0;
    }

    // one tensor per segment, produced by the same per-tensor allocation path a
    // publishing load uses (alloc_buffer_n on the shareable buffer type)
    const size_t ctx_size = ggml_tensor_overhead() * todo.size();
    ggml_context * ctx = ggml_init({ ctx_size, nullptr, true });

    std::vector<ggml_tensor *> tensors;
    tensors.reserve(todo.size());
    for (const work & w : todo) {
        const enum ggml_type type = gguf_get_tensor_type(gf, w.gguf_idx);
        const int64_t * ne = gguf_get_tensor_ne(gf, w.gguf_idx);
        ggml_tensor * t = ggml_new_tensor(ctx, type, GGML_MAX_DIMS, ne);
        ggml_set_name(t, w.name.c_str());
        if (ggml_nbytes(t) != w.nbytes) {
            fprintf(stderr, "prewarm: tensor '%s': created size %zu != manifest %zu, aborting\n",
                    w.name.c_str(), ggml_nbytes(t), w.nbytes);
            return 1;
        }
        tensors.push_back(t);
    }

    double t_alloc0 = now_s();
    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer_n(buft, tensors.data(), (int) tensors.size());
    if (buffer == nullptr) {
        fprintf(stderr, "prewarm: failed to allocate the segments\n");
        return 1;
    }
    printf("prewarm: allocated %zu segments at %.3f s\n", tensors.size(), now_s() - t_alloc0);

    // fill from the data section and register
    const double  t_s0    = now_s();
    const int     gguf_fd = open(gguf_path.c_str(), O_RDONLY);
    std::vector<char> chunk(16 * 1024 * 1024);
    if (gguf_fd < 0) {
        fprintf(stderr, "prewarm: cannot open %s\n", gguf_path.c_str());
        return 1;
    }

    size_t seeded_file = 0, seeded_seg = 0, registered = 0, failed = 0;
    for (size_t i = 0; i < tensors.size(); i++) {
        ggml_tensor * t = tensors[i];
        const size_t  n       = ggml_nbytes(t);
        const size_t  file_off = data_off + gguf_get_tensor_offset(gf, todo[i].gguf_idx);

        bool fill_ok = true;
        for (size_t off = 0; off < n; off += chunk.size()) {
            const size_t len = off + chunk.size() < n ? chunk.size() : n - off;
            size_t got = 0;
            while (fill_ok && got < len) {
                const ssize_t r = pread(gguf_fd, chunk.data() + got, len - got, (off_t) (file_off + off + got));
                if (r <= 0) {
                    if (r < 0 && errno == EINTR) {
                        continue;
                    }
                    fill_ok = false;
                    break;
                }
                got += (size_t) r;
            }
            if (!fill_ok) {
                break;
            }
            ggml_backend_tensor_set(t, chunk.data(), off, len);
        }
        if (!fill_ok) {
            fprintf(stderr, "prewarm: short read on '%s', skipping its registration\n", ggml_get_name(t));
            failed++;
            continue;
        }
        seeded_file += n;

        int    fd       = -1;
        size_t seg_size = 0;
        if (fn_export(t->buffer, &fd, &seg_size) != 0) {
            failed++;
            continue;
        }
        if (registry_register(socket_path, todo[i].digest.data(), bus, seg_size, ggml_get_name(t), fd)) {
            registered++;
            seeded_seg += seg_size;
        } else {
            failed++;
        }

        if ((i + 1) % 64 == 0 || i + 1 == tensors.size()) {
            printf("prewarm: %zu/%zu filled (%.1f MiB read, %.1f MiB registered segments)\n",
                   i + 1, tensors.size(), seeded_file / 1048576.0, seeded_seg / 1048576.0);
        }
    }
    printf("prewarm: seeded %zu tensors (%.1f MiB file bytes -> %.1f MiB segments) in %.3f s; "
           "%" PRIu64 " were already resident, %zu failures\n",
           registered, seeded_file / 1048576.0, seeded_seg / 1048576.0, now_s() - t_s0,
           resident, failed);

    // exit releases this process's mappings; the registry's fds keep the physical
    // allocations alive (that is the pin-survival property under test)
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    gguf_free(gf);
    close(gguf_fd);
    return registered == tensors.size() ? 0 : 1;
}