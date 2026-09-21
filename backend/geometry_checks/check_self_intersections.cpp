#include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#include <array>
#include <CGAL/Polygon_mesh_processing/self_intersections.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
using K=CGAL::Exact_predicates_exact_constructions_kernel;

int main(int argc,char**argv){
  if(argc!=3){std::cerr<<"mesh.obj diagonal(0|1)\n";return 2;}
  try{
    int diagonal=std::stoi(argv[2]);if(diagonal!=0&&diagonal!=1)return 2;
    std::ifstream file(argv[1]);if(!file)return 3;
    std::vector<K::Point_3> verts;std::vector<std::array<size_t,3>> triangles;std::vector<int> original;
    std::string line;int polygon=0,invalid=0,degenerate=0;
    while(std::getline(file,line)){
      std::istringstream stream(line);std::string kind;stream>>kind;
      if(kind=="v"){
        double x,y,z;if(!(stream>>x>>y>>z)||!std::isfinite(x)||!std::isfinite(y)||!std::isfinite(z))return 4;
        verts.emplace_back(x,y,z);
      }else if(kind=="f"){
        std::vector<int> face;std::string token;
        while(stream>>token){int v=std::stoi(token.substr(0,token.find('/')));v=v>0?v-1:int(verts.size())+v;if(v<0||v>=int(verts.size()))return 5;face.push_back(v);}
        if(face.size()!=3&&face.size()!=4)return 6;
        if(diagonal&&face.size()==4)std::rotate(face.begin(),face.begin()+1,face.end());
        for(size_t j=1;j+1<face.size();++j){
          size_t a=face[0],b=face[j],c=face[j+1];
          if(CGAL::collinear(verts[a],verts[b],verts[c]))++degenerate;
          triangles.push_back({a,b,c});original.push_back(polygon);
        }++polygon;
      }
    }
    if(invalid){std::cout<<"{\"completed\":false,\"invalid_triangle_insertions\":"<<invalid<<"}";return 0;}
    std::vector<std::pair<size_t,size_t>> pairs;
    CGAL::Polygon_mesh_processing::triangle_soup_self_intersections<CGAL::Sequential_tag>(verts,triangles,std::back_inserter(pairs));
    std::cout<<"{\"completed\":true,\"diagonal\":"<<diagonal<<",\"polygons\":"<<polygon<<",\"triangles\":"<<original.size()<<",\"exact_degenerate_triangles\":"<<degenerate<<",\"intersection_pair_count\":"<<pairs.size()<<",\"pairs\":[";
    for(size_t i=0;i<pairs.size();++i){if(i)std::cout<<",";auto a=pairs[i].first,b=pairs[i].second;std::cout<<"{\"triangles\":["<<a<<","<<b<<"],\"polygons\":["<<original[a]<<","<<original[b]<<"]}";}
    std::cout<<"]}";
  }catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 8;}
}
