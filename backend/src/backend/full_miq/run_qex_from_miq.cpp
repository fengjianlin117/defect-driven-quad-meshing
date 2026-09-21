#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <qex.h>
#include <Eigen/Core>
#include <igl/planarize_quad_mesh.h>

bool read_obj(const char *path, std::vector<double> &verts, std::vector<int> &faces) {
    FILE *fp = fopen(path, "r");
    if (!fp) { fprintf(stderr, "ERROR: Cannot open %s\n", path); return false; }
    char line[4096];
    while (fgets(line, sizeof(line), fp)) {
        if (line[0] == '\0' || line[0] == '#') continue;
        if (line[0] == 'v' && line[1] == ' ') {
            double x, y, z;
            if (sscanf(line, "v %lf %lf %lf", &x, &y, &z) == 3) {
                verts.push_back(x); verts.push_back(y); verts.push_back(z);
            }
        } else if (line[0] == 'f' && line[1] == ' ') {
            int a, b, c;
            if (sscanf(line, "f %d %d %d", &a, &b, &c) == 3) {
                faces.push_back(a-1); faces.push_back(b-1); faces.push_back(c-1);
            } else if (sscanf(line, "f %d/%*d %d/%*d %d/%*d", &a, &b, &c) >= 3) {
                faces.push_back(a-1); faces.push_back(b-1); faces.push_back(c-1);
            }
        }
    }
    fclose(fp);
    return !faces.empty();
}

bool read_mat(const char *path, std::vector<double> &data, int &rows, int &cols) {
    FILE *fp = fopen(path, "r");
    if (!fp) { fprintf(stderr, "ERROR: Cannot open %s\n", path); return false; }
    if (fscanf(fp, "%d %d", &rows, &cols) != 2) {
        fprintf(stderr, "ERROR: Invalid header\n"); fclose(fp); return false;
    }
    data.reserve(rows * cols);
    for (int i = 0; i < rows * cols; i++) {
        double val;
        if (fscanf(fp, "%lf", &val) != 1) {
            fprintf(stderr, "ERROR: Read error at %d\n", i); fclose(fp); return false;
        }
        data.push_back(val);
    }
    fclose(fp);
    return true;
}

int main(int argc, char *argv[]) {
    if (argc < 4) {
        fprintf(stderr, "Usage: run_qex_from_miq input_tri.obj miq_prefix output_quad.obj [planarize=0] [threshold=0.01] [max_iter=100]\n");
        return 1;
    }
    const char *in_obj = argv[1];
    std::string prefix(argv[2]);
    const char *out_obj = argv[3];
    const int planarize = (argc > 4) ? atoi(argv[4]) : 0;
    const double planarize_threshold = (argc > 5) ? atof(argv[5]) : 0.01;
    const int planarize_iter = (argc > 6) ? atoi(argv[6]) : 100;

    fprintf(stdout, "=== run_qex_from_miq ===\n");
    fprintf(stdout, "Input: %s\n", in_obj);
    fprintf(stdout, "Output: %s\n", out_obj);

    std::vector<double> V;
    std::vector<int> F;
    if (!read_obj(in_obj, V, F)) return 1;
    int nV = (int)V.size() / 3;
    int nF = (int)F.size() / 3;
    fprintf(stdout, "V=%d F=%d\n", nV, nF);

    std::string uv_path = prefix + "_uv.txt";
    std::string fuv_path = prefix + "_fuv.txt";

    std::vector<double> UV;
    int uv_rows = 0, uv_cols = 0;
    if (!read_mat(uv_path.c_str(), UV, uv_rows, uv_cols)) return 1;
    fprintf(stdout, "UV: %dx%d\n", uv_rows, uv_cols);

    std::vector<double> FUV_flat;
    int fuv_rows = 0, fuv_cols = 0;
    if (!read_mat(fuv_path.c_str(), FUV_flat, fuv_rows, fuv_cols)) return 1;
    fprintf(stdout, "FUV: %dx%d\n", fuv_rows, fuv_cols);

    if (uv_cols != 2) { fprintf(stderr, "UV must have 2 cols\n"); return 1; }
    if (fuv_cols != 3) { fprintf(stderr, "FUV must have 3 cols\n"); return 1; }
    if (fuv_rows != nF) {
        fprintf(stderr, "FUV rows %d != faces %d\n", fuv_rows, nF); return 1;
    }

    double u_min=1e30,u_max=-1e30,v_min=1e30,v_max=-1e30;
    bool has_nan = false;
    for (int i = 0; i < uv_rows; i++) {
        double u = UV[i*2], v = UV[i*2+1];
        if (std::isnan(u)||std::isinf(u)||std::isnan(v)||std::isinf(v)) has_nan = true;
        if (u < u_min) u_min = u; if (u > u_max) u_max = u;
        if (v < v_min) v_min = v; if (v > v_max) v_max = v;
    }
    fprintf(stdout, "UV range: u=[%.2f,%.2f] v=[%.2f,%.2f]\n", u_min, u_max, v_min, v_max);
    fprintf(stdout, "UV NaN: %s\n", has_nan ? "YES" : "No");

    int max_fuv = -1;
    for (int i = 0; i < fuv_rows * 3; i++) {
        int idx = (int)FUV_flat[i];
        if (idx > max_fuv) max_fuv = idx;
        if (FUV_flat[i] < 0) { fprintf(stderr, "FUV negative\n"); return 1; }
    }
    if (max_fuv >= uv_rows) {
        fprintf(stderr, "FUV index %d >= UV %d\n", max_fuv, uv_rows); return 1;
    }
    fprintf(stdout, "FUV max idx: %d / UV: %d\n", max_fuv, uv_rows);

    qex_TriMesh triMesh;
    triMesh.vertex_count = nV;
    triMesh.tri_count = nF;
    triMesh.vertices = (qex_Point3*)malloc(nV * sizeof(qex_Point3));
    triMesh.tris = (qex_Tri*)malloc(nF * sizeof(qex_Tri));
    triMesh.uvTris = (qex_UVTri*)malloc(nF * sizeof(qex_UVTri));

    for (int i = 0; i < nV; i++) {
        triMesh.vertices[i].x[0] = V[i*3];
        triMesh.vertices[i].x[1] = V[i*3+1];
        triMesh.vertices[i].x[2] = V[i*3+2];
    }
    for (int i = 0; i < nF; i++) {
        triMesh.tris[i].indices[0] = F[i*3];
        triMesh.tris[i].indices[1] = F[i*3+1];
        triMesh.tris[i].indices[2] = F[i*3+2];
        int f0 = (int)FUV_flat[i*3], f1 = (int)FUV_flat[i*3+1], f2 = (int)FUV_flat[i*3+2];
        triMesh.uvTris[i].uvs[0].x[0] = UV[f0*2];   triMesh.uvTris[i].uvs[0].x[1] = UV[f0*2+1];
        triMesh.uvTris[i].uvs[1].x[0] = UV[f1*2];   triMesh.uvTris[i].uvs[1].x[1] = UV[f1*2+1];
        triMesh.uvTris[i].uvs[2].x[0] = UV[f2*2];   triMesh.uvTris[i].uvs[2].x[1] = UV[f2*2+1];
    }
    fprintf(stdout, "qex_TriMesh: V=%u F=%u\n", triMesh.vertex_count, triMesh.tri_count);
    fprintf(stdout, "Calling qex_extractQuadMesh...\n");
    fflush(stdout);

    qex_QuadMesh quadMesh = {0, 0, NULL, NULL};
    qex_extractQuadMesh(&triMesh, NULL, &quadMesh);

    fprintf(stdout, "Output: %u vertices, %u quads\n", quadMesh.vertex_count, quadMesh.quad_count);

    Eigen::MatrixXd Vq(quadMesh.vertex_count, 3);
    Eigen::MatrixXi Fq(quadMesh.quad_count, 4);
    for (unsigned i = 0; i < quadMesh.vertex_count; i++)
        Vq.row(i) << quadMesh.vertices[i].x[0], quadMesh.vertices[i].x[1], quadMesh.vertices[i].x[2];
    for (unsigned i = 0; i < quadMesh.quad_count; i++)
        Fq.row(i) << quadMesh.quads[i].indices[0], quadMesh.quads[i].indices[1],
                     quadMesh.quads[i].indices[2], quadMesh.quads[i].indices[3];

    if (planarize) {
        Eigen::MatrixXd Vout;
        igl::planarize_quad_mesh(Vq, Fq, planarize_iter, planarize_threshold, Vout);
        Vq = Vout;
        fprintf(stdout, "Planarized: iter=%d threshold=%.3f\n", planarize_iter, planarize_threshold);
    }

    FILE *out = fopen(out_obj, "w");
    if (!out) { fprintf(stderr, "Cannot write %s\n", out_obj); return 1; }
    for (unsigned i = 0; i < (unsigned)Vq.rows(); i++)
        fprintf(out, "v %.15f %.15f %.15f\n",
                Vq(i,0), Vq(i,1), Vq(i,2));
    for (unsigned i = 0; i < (unsigned)Fq.rows(); i++)
        fprintf(out, "f %u %u %u %u\n",
                Fq(i,0)+1, Fq(i,1)+1, Fq(i,2)+1, Fq(i,3)+1);
    fclose(out);

    free(triMesh.vertices); free(triMesh.tris); free(triMesh.uvTris);
    free(quadMesh.vertices); free(quadMesh.quads);

    fprintf(stdout, "=== run_qex_from_miq done ===\n");
    return (quadMesh.quad_count > 0) ? 0 : 1;
}
