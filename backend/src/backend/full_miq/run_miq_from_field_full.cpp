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

static Edge canonicalEdge(int a, int b) {
    return a < b ? Edge(a, b) : Edge(b, a);
}

static MatrixXd readDirectionField(const char *path, int faceCount) {
    FILE *file = std::fopen(path, "r");
    if (!file) {
        std::fprintf(stderr, "ERROR: cannot open %s\n", path);
        std::exit(1);
    }
    MatrixXd field = MatrixXd::Zero(faceCount, 3);
    std::vector<bool> seen(faceCount, false);
    char line[4096];
    int id;
    double x, y, z;
    while (std::fgets(line, sizeof(line), file)) {
        if (line[0] == '#' || line[0] == '\n' || line[0] == '\r') continue;
        if (std::sscanf(line, "%d %lf %lf %lf", &id, &x, &y, &z) != 4 ||
            id < 0 || id >= faceCount || seen[id]) {
            std::fprintf(stderr, "ERROR: invalid or duplicate row in %s\n", path);
            std::fclose(file);
            std::exit(1);
        }
        seen[id] = true;
        field.row(id) = RowVector3d(x, y, z);
    }
    std::fclose(file);
    for (int face = 0; face < faceCount; ++face) {
        if (!seen[face] || field.row(face).norm() <= 1e-12) {
            std::fprintf(stderr, "ERROR: missing or zero direction for face %d in %s\n", face, path);
            std::exit(1);
        }
    }
    return field;
}

static std::set<Edge> loadStructuralEdges(
    const char *path, int vertexCount, const MatrixXi &faces
) {
    std::set<Edge> meshEdges;
    for (int face = 0; face < faces.rows(); ++face)
        for (int local = 0; local < 3; ++local)
            meshEdges.insert(canonicalEdge(faces(face, local), faces(face, (local + 1) % 3)));

    FILE *file = std::fopen(path, "r");
    if (!file) {
        std::fprintf(stderr, "ERROR: cannot open structural edges %s\n", path);
        std::exit(1);
    }
    std::set<Edge> edges;
    char line[4096];
    int lineNumber = 0;
    while (std::fgets(line, sizeof(line), file)) {
        ++lineNumber;
        if (line[0] == '#' || line[0] == '\n' || line[0] == '\r') continue;
        int a, b;
        char extra;
        if (std::sscanf(line, "%d %d %c", &a, &b, &extra) != 2 ||
            a < 0 || b < 0 || a >= vertexCount || b >= vertexCount || a == b) {
            std::fprintf(stderr, "ERROR: invalid structural edge at line %d\n", lineNumber);
            std::fclose(file);
            std::exit(1);
        }
        const Edge edge = canonicalEdge(a, b);
        if (!meshEdges.count(edge) || !edges.insert(edge).second) {
            std::fprintf(stderr, "ERROR: unresolved or duplicate structural edge (%d, %d)\n", edge.first, edge.second);
            std::fclose(file);
            std::exit(1);
        }
    }
    std::fclose(file);
    return edges;
}

int main(int argc, char *argv[]) {
    if (argc == 2 && std::strcmp(argv[1], "--capabilities") == 0) {
        std::puts("{\"schema\":\"miq-adapter.v2\",\"density_aware\":false,\"stiffness_iterations\":true}");
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
    if (argc < 6 || argc > 7) {
        std::fprintf(stderr,
            "Usage: run_miq_from_field_full input.obj PD1.txt PD2.txt output_prefix gsize [structural_edges.txt]\n");
        return 1;
    }

    MatrixXd V;
    MatrixXi F;
    if (!igl::readOBJ(argv[1], V, F)) {
        std::fprintf(stderr, "ERROR: failed to read OBJ %s\n", argv[1]);
        return 1;
    }
    char *gsizeEnd = nullptr;
    errno = 0;
    const double gSize = std::strtod(argv[5], &gsizeEnd);
    if (errno || gsizeEnd == argv[5] || *gsizeEnd || !std::isfinite(gSize) || gSize <= 0.0) {
        std::fprintf(stderr, "ERROR: gsize must be a finite positive number\n");
        return 1;
    }
    const int faceCount = F.rows();
    MatrixXd PD1 = readDirectionField(argv[2], faceCount);
    MatrixXd PD2 = readDirectionField(argv[3], faceCount);

    for (int face = 0; face < faceCount; ++face) {
        const RowVector3d e1 = V.row(F(face, 1)) - V.row(F(face, 0));
        const RowVector3d e2 = V.row(F(face, 2)) - V.row(F(face, 0));
        Vector3d normal = e1.cross(e2);
        if (normal.norm() <= 1e-12) {
            std::fprintf(stderr, "ERROR: degenerate triangle at face %d\n", face);
            return 1;
        }
        normal.normalize();
        const RowVector3d normalRow = normal.transpose();
        RowVector3d first = PD1.row(face) - PD1.row(face).dot(normalRow) * normalRow;
        if (first.norm() <= 1e-12) {
            std::fprintf(stderr, "ERROR: PD1 is normal to face %d\n", face);
            return 1;
        }
        first.normalize();
        PD1.row(face) = first;
        PD2.row(face) = normal.cross(first.transpose()).normalized().transpose();
    }

    std::set<Edge> structuralEdges;
    std::vector<std::vector<int>> hardFeatures;
    if (argc == 7) {
        structuralEdges = loadStructuralEdges(argv[6], V.rows(), F);
        for (int face = 0; face < faceCount; ++face)
            for (int local = 0; local < 3; ++local)
                if (structuralEdges.count(canonicalEdge(F(face, local), F(face, (local + 1) % 3))))
                    hardFeatures.push_back({face, local});

        char path[512];
        std::snprintf(path, sizeof(path), "%s_resolved_structural_edges.txt", argv[4]);
        FILE *output = std::fopen(path, "w");
        for (const Edge &edge : structuralEdges) std::fprintf(output, "%d %d\n", edge.first, edge.second);
        std::fclose(output);
        std::snprintf(path, sizeof(path), "%s_resolved_hard_features.tsv", argv[4]);
        output = std::fopen(path, "w");
        std::fprintf(output, "face_id\tlocal_edge_id\tu\tv\n");
        for (const auto &feature : hardFeatures) {
            const Edge edge = canonicalEdge(F(feature[0], feature[1]), F(feature[0], (feature[1] + 1) % 3));
            std::fprintf(output, "%d\t%d\t%d\t%d\n", feature[0], feature[1], edge.first, edge.second);
        }
        std::fclose(output);
    }
    std::fprintf(stdout, "V=%d F=%d gsize=%.6g structural_edges=%zu hard_features=%zu\n",
        static_cast<int>(V.rows()), faceCount, gSize, structuralEdges.size(), hardFeatures.size());

    MatrixXd BIS1, BIS2, BIS1c, BIS2c, X1c, X2c;
    Matrix<int, Dynamic, 3> MMatch, Seams;
    Matrix<int, Dynamic, 1> isSing, singIdx;
    igl::compute_frame_field_bisectors(V, F, PD1, PD2, BIS1, BIS2);
    igl::comb_cross_field(V, F, BIS1, BIS2, BIS1c, BIS2c);
    igl::cross_field_mismatch(V, F, BIS1c, BIS2c, true, MMatch);
    igl::find_cross_field_singularities(V, F, MMatch, isSing, singIdx);
    igl::cut_mesh_from_singularities(V, F, MMatch, Seams);
    igl::comb_frame_field(V, F, PD1, PD2, BIS1c, BIS2c, X1c, X2c);

    // MIQ consumes the topology recomputed from the final combed frame.
    igl::compute_frame_field_bisectors(V, F, X1c, X2c, BIS1, BIS2);
    igl::comb_cross_field(V, F, BIS1, BIS2, BIS1c, BIS2c);
    igl::cross_field_mismatch(V, F, BIS1c, BIS2c, true, MMatch);
    igl::find_cross_field_singularities(V, F, MMatch, isSing, singIdx);
    igl::cut_mesh_from_singularities(V, F, MMatch, Seams);

    MatrixXd UV;
    MatrixXi FUV;
    if (hardFeatures.empty()) {
        igl::copyleft::comiso::miq(
            V, F, X1c, X2c, MMatch, isSing, Seams, UV, FUV, gSize, 5.0, false, stiffnessIterations);
    } else {
        igl::copyleft::comiso::miq(
            V, F, X1c, X2c, MMatch, isSing, Seams, UV, FUV,
            gSize, 5.0, false, stiffnessIterations, 5, true, true, std::vector<int>(), hardFeatures);
    }

    char path[512];
    std::snprintf(path, sizeof(path), "%s_uv.txt", argv[4]);
    { FILE *f = std::fopen(path, "w"); std::fprintf(f, "%d 2\n", static_cast<int>(UV.rows()));
      for (int i = 0; i < UV.rows(); ++i) std::fprintf(f, "%.15f %.15f\n", UV(i, 0), UV(i, 1)); std::fclose(f); }
    std::snprintf(path, sizeof(path), "%s_fuv.txt", argv[4]);
    { FILE *f = std::fopen(path, "w"); std::fprintf(f, "%d 3\n", static_cast<int>(FUV.rows()));
      for (int i = 0; i < FUV.rows(); ++i) std::fprintf(f, "%d %d %d\n", FUV(i, 0), FUV(i, 1), FUV(i, 2)); std::fclose(f); }
    std::snprintf(path, sizeof(path), "%s_combed_PD1.txt", argv[4]);
    { FILE *f = std::fopen(path, "w"); for (int i = 0; i < X1c.rows(); ++i)
      std::fprintf(f, "%d %.15f %.15f %.15f\n", i, X1c(i, 0), X1c(i, 1), X1c(i, 2)); std::fclose(f); }
    std::snprintf(path, sizeof(path), "%s_combed_PD2.txt", argv[4]);
    { FILE *f = std::fopen(path, "w"); for (int i = 0; i < X2c.rows(); ++i)
      std::fprintf(f, "%d %.15f %.15f %.15f\n", i, X2c(i, 0), X2c(i, 1), X2c(i, 2)); std::fclose(f); }
    std::snprintf(path, sizeof(path), "%s_mismatch.txt", argv[4]);
    { FILE *f = std::fopen(path, "w"); for (int i = 0; i < MMatch.rows(); ++i)
      std::fprintf(f, "%d %d %d\n", MMatch(i, 0), MMatch(i, 1), MMatch(i, 2)); std::fclose(f); }
    std::snprintf(path, sizeof(path), "%s_singularities.txt", argv[4]);
    { FILE *f = std::fopen(path, "w"); for (int i = 0; i < isSing.size(); ++i)
      std::fprintf(f, "%d %d\n", i, isSing(i)); std::fclose(f); }
    std::snprintf(path, sizeof(path), "%s_singularities_detailed.txt", argv[4]);
    { FILE *f = std::fopen(path, "w"); for (int i = 0; i < isSing.size(); ++i)
      std::fprintf(f, "%d %d %d\n", i, isSing(i), singIdx(i)); std::fclose(f); }
    std::snprintf(path, sizeof(path), "%s_seams.txt", argv[4]);
    { FILE *f = std::fopen(path, "w"); for (int i = 0; i < Seams.rows(); ++i)
      std::fprintf(f, "%d %d %d\n", Seams(i, 0), Seams(i, 1), Seams(i, 2)); std::fclose(f); }

    std::fprintf(stdout, "MIQ complete: UV=%d FUV=%d\n", static_cast<int>(UV.rows()), static_cast<int>(FUV.rows()));
    return 0;
}
