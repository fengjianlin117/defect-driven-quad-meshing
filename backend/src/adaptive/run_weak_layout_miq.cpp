#include <cerrno>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <set>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <igl/comb_cross_field.h>
#include <igl/comb_frame_field.h>
#include <igl/compute_frame_field_bisectors.h>
#include <igl/copyleft/comiso/miq.h>
#include <igl/cross_field_mismatch.h>
#include <igl/cut_mesh_from_singularities.h>
#include <igl/find_cross_field_singularities.h>
#include <igl/readOBJ.h>

using namespace Eigen;
using Edge = std::pair<int, int>;

static Edge canonicalEdge(int a, int b) { return a < b ? Edge(a, b) : Edge(b, a); }

static MatrixXd readDirectionField(const char *path, int faceCount) {
    FILE *file = std::fopen(path, "r");
    if (!file) { std::fprintf(stderr, "ERROR: cannot open %s\n", path); std::exit(2); }
    MatrixXd field = MatrixXd::Zero(faceCount, 3);
    std::vector<bool> seen(faceCount, false);
    char line[4096]; int id; double x, y, z;
    while (std::fgets(line, sizeof(line), file)) {
        if (line[0] == '#' || line[0] == '\n' || line[0] == '\r') continue;
        if (std::sscanf(line, "%d %lf %lf %lf", &id, &x, &y, &z) != 4 ||
            id < 0 || id >= faceCount || seen[id] || !std::isfinite(x + y + z)) {
            std::fprintf(stderr, "ERROR: invalid/duplicate direction row in %s\n", path);
            std::fclose(file); std::exit(2);
        }
        seen[id] = true; field.row(id) = RowVector3d(x, y, z);
    }
    std::fclose(file);
    for (int face = 0; face < faceCount; ++face)
        if (!seen[face] || field.row(face).norm() <= 1e-12) {
            std::fprintf(stderr, "ERROR: missing/zero direction for face %d in %s\n", face, path);
            std::exit(2);
        }
    return field;
}

static VectorXd readDensity(const char *path, int faceCount) {
    FILE *file = std::fopen(path, "r");
    if (!file) { std::fprintf(stderr, "ERROR: cannot open density %s\n", path); std::exit(2); }
    VectorXd density = VectorXd::Zero(faceCount);
    std::vector<bool> seen(faceCount, false);
    char line[4096]; int sequential = 0;
    while (std::fgets(line, sizeof(line), file)) {
        if (line[0] == '#' || line[0] == '\n' || line[0] == '\r') continue;
        int id; double value; char first[256], second[256], extra[256];
        const int tokens = std::sscanf(line, "%255s %255s %255s", first, second, extra);
        char *valueEnd = nullptr;
        if (tokens == 2) {
            char *idEnd = nullptr;
            const long parsedId = std::strtol(first, &idEnd, 10);
            value = std::strtod(second, &valueEnd);
            if (!idEnd || *idEnd || !valueEnd || *valueEnd) {
                std::fprintf(stderr, "ERROR: invalid indexed density row\n"); std::exit(2);
            }
            id = static_cast<int>(parsedId);
            if (id < 0 || id >= faceCount || seen[id]) {
                std::fprintf(stderr, "ERROR: invalid/duplicate density id %d\n", id); std::exit(2);
            }
        } else if (tokens == 1) {
            value = std::strtod(first, &valueEnd);
            if (!valueEnd || *valueEnd) { std::fprintf(stderr, "ERROR: invalid density value\n"); std::exit(2); }
            while (sequential < faceCount && seen[sequential]) ++sequential;
            id = sequential++;
        } else {
            std::fprintf(stderr, "ERROR: invalid density row in %s\n", path); std::exit(2);
        }
        if (id >= faceCount || !std::isfinite(value) || value <= 0.0) {
            std::fprintf(stderr, "ERROR: density must be finite and positive\n"); std::exit(2);
        }
        density(id) = value; seen[id] = true;
    }
    std::fclose(file);
    for (int face = 0; face < faceCount; ++face)
        if (!seen[face]) { std::fprintf(stderr, "ERROR: missing density for face %d\n", face); std::exit(2); }
    return density;
}

static std::set<Edge> loadHardEdges(const char *path, int vertexCount, const MatrixXi &faces) {
    std::set<Edge> meshEdges;
    for (int face = 0; face < faces.rows(); ++face)
        for (int local = 0; local < 3; ++local)
            meshEdges.insert(canonicalEdge(faces(face, local), faces(face, (local + 1) % 3)));
    FILE *file = std::fopen(path, "r");
    if (!file) { std::fprintf(stderr, "ERROR: cannot open hard edges %s\n", path); std::exit(2); }
    std::set<Edge> edges; char line[4096]; int number = 0;
    while (std::fgets(line, sizeof(line), file)) {
        ++number;
        if (line[0] == '#' || line[0] == '\n' || line[0] == '\r') continue;
        int a, b; char extra;
        if (std::sscanf(line, "%d %d %c", &a, &b, &extra) != 2 || a < 0 || b < 0 ||
            a >= vertexCount || b >= vertexCount || a == b) {
            std::fprintf(stderr, "ERROR: invalid hard edge at line %d\n", number); std::exit(2);
        }
        Edge edge = canonicalEdge(a, b);
        if (!meshEdges.count(edge) || !edges.insert(edge).second) {
            std::fprintf(stderr, "ERROR: unresolved/duplicate hard edge (%d,%d)\n", edge.first, edge.second);
            std::exit(2);
        }
    }
    std::fclose(file); return edges;
}

template <typename Derived>
static void writeMatrix(const char *path, const Eigen::MatrixBase<Derived> &matrix, bool header) {
    FILE *file = std::fopen(path, "w");
    if (!file) { std::fprintf(stderr, "ERROR: cannot write %s\n", path); std::exit(3); }
    if (header) std::fprintf(file, "%d %d\n", static_cast<int>(matrix.rows()), static_cast<int>(matrix.cols()));
    for (int row = 0; row < matrix.rows(); ++row) {
        for (int col = 0; col < matrix.cols(); ++col) {
            if (col) std::fputc(' ', file);
            std::fprintf(file, "%.15g", static_cast<double>(matrix(row, col)));
        }
        std::fputc('\n', file);
    }
    std::fclose(file);
}

int main(int argc, char *argv[]) {
    if (argc == 2 && std::strcmp(argv[1], "--capabilities") == 0) {
        std::puts("{\"schema\":\"miq-adapter.v2\",\"density_aware\":true,\"stiffness_iterations\":true}");
        return 0;
    }
    unsigned int stiffnessIterations = 0;
    if (argc >= 3 && std::strcmp(argv[argc-2], "--stiffness-iterations") == 0) {
        char *endIter = nullptr;
        const long parsed = std::strtol(argv[argc-1], &endIter, 10);
        if (endIter == argv[argc-1] || *endIter || parsed < 0 || parsed > 100) {
            std::fprintf(stderr, "ERROR: stiffness iterations must be in [0,100]\n"); return 2;
        }
        stiffnessIterations = static_cast<unsigned int>(parsed); argc -= 2;
    }
    if (argc < 7 || argc > 8) {
        std::fprintf(stderr, "Usage: run_weak_layout_miq mesh.obj PD1.txt PD2.txt face_density.txt output_prefix gsize [hard_feature_edges.txt]\n");
        return 2;
    }
    MatrixXd V; MatrixXi F;
    if (!igl::readOBJ(argv[1], V, F)) { std::fprintf(stderr, "ERROR: failed to read %s\n", argv[1]); return 2; }
    char *end = nullptr; errno = 0; const double gsize = std::strtod(argv[6], &end);
    if (errno || end == argv[6] || *end || !std::isfinite(gsize) || gsize <= 0) {
        std::fprintf(stderr, "ERROR: gsize must be finite and positive\n"); return 2;
    }
    MatrixXd PD1 = readDirectionField(argv[2], F.rows());
    MatrixXd PD2 = readDirectionField(argv[3], F.rows());
    VectorXd density = readDensity(argv[4], F.rows());
    for (int face = 0; face < F.rows(); ++face) {
        const RowVector3d edge1 = V.row(F(face, 1)) - V.row(F(face, 0));
        const RowVector3d edge2 = V.row(F(face, 2)) - V.row(F(face, 0));
        Vector3d normal = edge1.cross(edge2);
        if (normal.norm() <= 1e-12) { std::fprintf(stderr, "ERROR: degenerate face %d\n", face); return 2; }
        normal.normalize(); RowVector3d n = normal.transpose();
        RowVector3d first = PD1.row(face) - PD1.row(face).dot(n) * n;
        if (first.norm() <= 1e-12) { std::fprintf(stderr, "ERROR: invalid PD1 face %d\n", face); return 2; }
        first.normalize(); PD1.row(face) = first;
        // Match the frozen backend convention exactly for rho=1.
        PD2.row(face) = normal.cross(first.transpose()).normalized().transpose();
    }
    std::set<Edge> hardEdges;
    std::vector<std::vector<int>> hardFeatures;
    if (argc == 8) {
        hardEdges = loadHardEdges(argv[7], V.rows(), F);
        for (int face = 0; face < F.rows(); ++face)
            for (int local = 0; local < 3; ++local)
                if (hardEdges.count(canonicalEdge(F(face, local), F(face, (local + 1) % 3))))
                    hardFeatures.push_back({face, local});
    }
    std::fprintf(stdout, "V=%d F=%d gsize=%.9g density=[%.9g,%.9g] hard_edges=%zu hard_features=%zu\n",
        static_cast<int>(V.rows()), static_cast<int>(F.rows()), gsize, density.minCoeff(), density.maxCoeff(),
        hardEdges.size(), hardFeatures.size());

    MatrixXd BIS1, BIS2, BIS1c, BIS2c, X1c, X2c;
    Matrix<int, Dynamic, 3> mismatch, seams;
    Matrix<int, Dynamic, 1> singular, singularIndex;
    igl::compute_frame_field_bisectors(V, F, PD1, PD2, BIS1, BIS2);
    igl::comb_cross_field(V, F, BIS1, BIS2, BIS1c, BIS2c);
    igl::cross_field_mismatch(V, F, BIS1c, BIS2c, true, mismatch);
    igl::find_cross_field_singularities(V, F, mismatch, singular, singularIndex);
    igl::cut_mesh_from_singularities(V, F, mismatch, seams);
    igl::comb_frame_field(V, F, PD1, PD2, BIS1c, BIS2c, X1c, X2c);
    // Density enters MIQ as a spatially varying target gradient. Topological
    // quantities are intentionally computed before scaling, preserving combing,
    // mismatch, singularities and seamless integer constraints.
    for (int face = 0; face < F.rows(); ++face) {
        X1c.row(face) *= density(face);
        X2c.row(face) *= density(face);
    }
    igl::compute_frame_field_bisectors(V, F, X1c, X2c, BIS1, BIS2);
    igl::comb_cross_field(V, F, BIS1, BIS2, BIS1c, BIS2c);
    igl::cross_field_mismatch(V, F, BIS1c, BIS2c, true, mismatch);
    igl::find_cross_field_singularities(V, F, mismatch, singular, singularIndex);
    igl::cut_mesh_from_singularities(V, F, mismatch, seams);

    MatrixXd UV; MatrixXi FUV;
    if (hardFeatures.empty())
        igl::copyleft::comiso::miq(V, F, X1c, X2c, mismatch, singular, seams, UV, FUV, gsize, 5.0, false, stiffnessIterations);
    else
        igl::copyleft::comiso::miq(V, F, X1c, X2c, mismatch, singular, seams, UV, FUV,
            gsize, 5.0, false, stiffnessIterations, 5, true, true, std::vector<int>(), hardFeatures);

    char path[4096];
    std::snprintf(path, sizeof(path), "%s_uv.txt", argv[5]); writeMatrix(path, UV, true);
    std::snprintf(path, sizeof(path), "%s_fuv.txt", argv[5]); writeMatrix(path, FUV, true);
    std::snprintf(path, sizeof(path), "%s_combed_PD1.txt", argv[5]); writeMatrix(path, X1c, false);
    std::snprintf(path, sizeof(path), "%s_combed_PD2.txt", argv[5]); writeMatrix(path, X2c, false);
    std::snprintf(path, sizeof(path), "%s_mismatch.txt", argv[5]); writeMatrix(path, mismatch, false);
    std::snprintf(path, sizeof(path), "%s_seams.txt", argv[5]); writeMatrix(path, seams, false);
    MatrixXi singularRows(singular.rows(), 3);
    for (int i = 0; i < singular.rows(); ++i) singularRows.row(i) << i, singular(i), singularIndex(i);
    std::snprintf(path, sizeof(path), "%s_singularities_detailed.txt", argv[5]); writeMatrix(path, singularRows, false);
    std::snprintf(path, sizeof(path), "%s_resolved_hard_features.tsv", argv[5]);
    FILE *features = std::fopen(path, "w"); std::fprintf(features, "face_id\tlocal_edge_id\tu\tv\n");
    for (const auto &feature : hardFeatures) {
        Edge edge = canonicalEdge(F(feature[0], feature[1]), F(feature[0], (feature[1] + 1) % 3));
        std::fprintf(features, "%d\t%d\t%d\t%d\n", feature[0], feature[1], edge.first, edge.second);
    }
    std::fclose(features);
    std::fprintf(stdout, "MIQ complete: UV=%d FUV=%d\n", static_cast<int>(UV.rows()), static_cast<int>(FUV.rows()));
    return 0;
}
