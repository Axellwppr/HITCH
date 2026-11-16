// D455 capture worker: newline request -> JSON header + packed RGB8 + float32 axial depth.
// Links against librealsense (an RSUSB build works); no Python binding needed.
#include <librealsense2/rs.hpp>
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <limits>
#include <string>

int main(int argc, char** argv) try {
    if (argc != 3) throw std::runtime_error("capture_rgbd SERIAL DEPTH_HEIGHT");
    int dh = std::stoi(argv[2]);
    if (dh != 360 && dh != 480) throw std::runtime_error("depth height must be 360 or 480");
    rs2::pipeline pipe;
    rs2::config cfg;
    cfg.enable_device(argv[1]);
    cfg.enable_stream(RS2_STREAM_COLOR, 1280, 800, RS2_FORMAT_RGB8, 30);
    cfg.enable_stream(RS2_STREAM_DEPTH, 640, dh, RS2_FORMAT_Z16, 30);
    auto profile = pipe.start(cfg);
    for (auto sensor : profile.get_device().query_sensors())
        if (sensor.supports(RS2_OPTION_FRAMES_QUEUE_SIZE)) sensor.set_option(RS2_OPTION_FRAMES_QUEUE_SIZE, 1);
    auto intr = profile.get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>().get_intrinsics();
    const char* distortion = nullptr;
    if (intr.model == RS2_DISTORTION_BROWN_CONRADY) distortion = "brown_conrady";
    else if (intr.model == RS2_DISTORTION_MODIFIED_BROWN_CONRADY) distortion = "modified_brown_conrady";
    else if (intr.model == RS2_DISTORTION_INVERSE_BROWN_CONRADY) distortion = "inverse_brown_conrady";
    else if (intr.model != RS2_DISTORTION_NONE) throw std::runtime_error("unsupported color distortion");
    rs2::align align(RS2_STREAM_COLOR);
    for (int i=0; i<30; ++i) pipe.wait_for_frames(2000);
    std::string command;
    std::cout << std::setprecision(12);
    while (std::getline(std::cin, command)) {
        if (command != "capture" && command != "capture_native")
            throw std::runtime_error("unknown capture command");
        const int step = command == "capture_native" ? 1 : 2;
        const int w = intr.width / step, h = intr.height / step;
        const float fx = intr.fx / step, fy = intr.fy / step, cx = intr.ppx / step, cy = intr.ppy / step;
        // RGB and aligned depth use identical source pixels, preserving registration.
        // Discard buffered images and two new frames after the settled-head request.
        rs2::frameset frames;
        for (int i=0; i<8 && pipe.poll_for_frames(&frames); ++i) {}
        for (int i=0; i<2; ++i) frames = pipe.wait_for_frames(2000);
        auto color = frames.get_color_frame();
        auto depth = align.process(frames).as<rs2::frameset>().get_depth_frame();
        if (!color || !depth) throw std::runtime_error("missing aligned RGB-D");
        const auto* c = static_cast<const uint8_t*>(color.get_data());
        const auto* d = static_cast<const uint16_t*>(depth.get_data());
        std::vector<uint8_t> rgb(w*h*3, 0);
        std::vector<float> metres(w*h, std::numeric_limits<float>::quiet_NaN());
        for (int y=0; y<h; ++y) for (int x=0; x<w; ++x) {
            int i=y*w+x;
            for (int k=0; k<3; ++k) rgb[i*3+k]=c[step*y*color.get_stride_in_bytes()+step*x*3+k];
            float z=d[step*y*depth.get_stride_in_bytes()/2+step*x]*depth.get_units();
            if (z>0 && z<5) metres[i]=z;
        }
        std::cout << "{\"width\":" << w << ",\"height\":" << h << ",\"intrinsics\":{\"fx\":" << fx
                  << ",\"fy\":" << fy << ",\"cx\":" << cx << ",\"cy\":" << cy;
        if (distortion) {
            std::cout << ",\"distortion\":{\"model\":\"" << distortion << "\",\"coeffs\":[";
            for (int j=0; j<5; ++j) std::cout << (j ? "," : "") << intr.coeffs[j];
            std::cout << "]}";
        }
        std::cout << "},\"frame_id\":" << color.get_frame_number()
                  << ",\"color_timestamp_ms\":" << color.get_timestamp()
                  << ",\"depth_timestamp_ms\":" << depth.get_timestamp()
                  << ",\"timestamp_domain\":\"" << rs2_timestamp_domain_to_string(color.get_frame_timestamp_domain())
                  << "\",\"rectified\":false}\n";
        std::cout.write(reinterpret_cast<const char*>(rgb.data()), rgb.size());
        std::cout.write(reinterpret_cast<const char*>(metres.data()), metres.size()*sizeof(float));
        std::cout.flush();
        if (!std::cout) break;
    }
    pipe.stop();
    return 0;
} catch (const std::exception& e) { std::cerr << e.what() << std::endl; return 1; }
