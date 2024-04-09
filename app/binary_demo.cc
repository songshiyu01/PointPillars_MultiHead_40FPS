#include <fstream>
#include <experimental/filesystem>
#include <pcl/io/pcd_io.h>
#include "pointpillars.h"

void boxes_to_txt(const std::vector<float>& bb_boxes, 
    const std::experimental::filesystem::path& out_file, 
    int num_feature = 7) {
    ofstream ofFile(out_file, std::ios::out);  
    if (ofFile.is_open()) {
        for (int i = 0 ; i < bb_boxes.size() / num_feature ; ++i) {
            for (int j = 0 ; j < num_feature ; ++j) {
                ofFile << bb_boxes.at(i * num_feature + j) << " ";
            }
            ofFile << std::endl;
        }
    }
    ofFile.close();
    return;
}

int main(int argc, char **argv) {
  if (argc < 3)
  {
    std::cerr << "usage: binary_demo path_to_bootstrap.yaml path_to_bin_folder" << std::endl;
    return -1;
  }
  YAML::Node config = YAML::LoadFile(argv[1]);

  std::string data_dir(argv[2]);
  if (!std::experimental::filesystem::is_directory(data_dir))
  {
    std::cerr << data_dir << " is not a valid directory." << std::endl;
    return -1;
  }

  std::string pfe_file, backbone_file; 
  if(config["UseOnnx"].as<bool>()) {
    pfe_file = config["PfeOnnx"].as<std::string>();
    backbone_file = config["BackboneOnnx"].as<std::string>();
  }else {
    pfe_file = config["PfeTrt"].as<std::string>();
    backbone_file = config["BackboneTrt"].as<std::string>();
  }
  std::cout << pfe_file << std::endl;
  std::cout << backbone_file << std::endl;
  const std::string pp_config = config["ModelConfig"].as<std::string>();
  std::cout << pp_config << std::endl;
  PointPillars pp(
    config["ScoreThreshold"].as<float>(),
    config["NmsOverlapThreshold"].as<float>(),
    config["UseOnnx"].as<bool>(),
    pfe_file,
    backbone_file,
    pp_config
  );

  for(const std::experimental::filesystem::directory_entry& bin_file : 
      std::experimental::filesystem::directory_iterator(data_dir)) {
    if(bin_file.path().extension() == ".pcd") {
      std::ifstream pc_stream(bin_file.path(), std::ifstream::binary);
      if (pc_stream) {
        int kNumPointFeature = 5;
        // pc_stream.seekg(0, pc_stream.end);
        // int file_length = pc_stream.tellg();
        // std::vector<float> buf(file_length / sizeof(float));
        // pc_stream.read(reinterpret_cast<char*>(buf.data()), buf.size()*sizeof(float));
        // int num_of_points = buf.size() / kNumPointFeature;
        // assert(buf.size() % 5 == 0);
        pcl::PointCloud<pcl::PointXYZI>::Ptr pcl_pc(new pcl::PointCloud<pcl::PointXYZI>);
        if (pcl::io::loadPCDFile<pcl::PointXYZI>(bin_file.path().c_str(), *pcl_pc) == -1) {
          std::cerr << "error: failed to load " << bin_file << std::endl;
          return -1;
        }
        int num_of_points = pcl_pc->size();
        std::vector<float> buf(kNumPointFeature* num_of_points);
        for (int i = 0; i < num_of_points; i++) {
          buf[i*kNumPointFeature + 0] = pcl_pc->points[i].x;
          buf[i*kNumPointFeature + 1] = pcl_pc->points[i].y;
          buf[i*kNumPointFeature + 2] = pcl_pc->points[i].z;
          buf[i*kNumPointFeature + 3] = pcl_pc->points[i].intensity;
          buf[i*kNumPointFeature + 4] = 0.0;
        }        
        std::cout << "Processing " << num_of_points << " points in " << bin_file.path() << "." << std::endl;

        std::vector<float> out_detections;
        std::vector<int> out_labels;
        std::vector<float> out_scores;

        cudaDeviceSynchronize();
        pp.DoInference(buf.data(), num_of_points, &out_detections, &out_labels , &out_scores);
        cudaDeviceSynchronize();
        int BoxFeature = 7;
        int num_objects = out_detections.size() / BoxFeature;

        std::experimental::filesystem::path out_file = bin_file;
        out_file.replace_extension("txt");
        boxes_to_txt(out_detections, out_file);
        std::cout << "Saved " << out_detections.size() << " bounding boxes in " << out_file << "." << std::endl;
        pc_stream.close();
      }
    }
  }
  return 0;
}