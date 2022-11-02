#include ".vscode/eigen_fix.h"
#include "Triangle.hpp"
#include "rasterizer.hpp"
#include <eigen3/Eigen/Eigen>
#include <iostream>
#include <opencv2/opencv.hpp>

constexpr double MY_PI = 3.1415926;

Eigen::Matrix4f get_view_matrix(Eigen::Vector3f eye_pos)
{
    Eigen::Matrix4f view = Eigen::Matrix4f::Identity();

    Eigen::Matrix4f translate;
    translate << 1, 0, 0, -eye_pos[0], 0, 1, 0, -eye_pos[1], 0, 0, 1,
        -eye_pos[2], 0, 0, 0, 1;

    view = translate * view;

    return view;
}

inline float deg2rad(float deg)
{
    return MY_PI / 180 * deg;
}

Eigen::Matrix4f get_model_matrix(float rotation_angle)
{
    auto rad = deg2rad(rotation_angle);
    float angle_sin = sin(rad);
    float angle_cos = cos(rad);

    // Create the model matrix for rotating the triangle around the Z axis.
    Eigen::Matrix4f model;
    model << angle_cos, -angle_sin, 0, 0,
        angle_sin, angle_cos, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1;

    return model;
}

Eigen::Matrix4f get_projection_matrix(float eye_fov, float aspect_ratio,
                                      float zNear, float zFar)
{
    // Create the projection matrix for the given parameters.
    Eigen::Matrix4f perspective;
    perspective << zNear, 0, 0, 0,
        0, zNear, 0, 0,
        0, 0, zNear + zFar, zNear * zFar,
        0, 0, -1, 0;

    Eigen::Matrix4f orthogonal_move;

    orthogonal_move << 1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0.5 * (zNear + zFar),
        0, 0, 0, 1;

    auto fov = deg2rad(eye_fov);
    auto fov_2 = fov / 2.0f;
    auto t = zNear * tan(fov_2);
    auto r = t * aspect_ratio;

    Eigen::Matrix4f orthogonal_scale;
    orthogonal_scale << 1.0f / r, 0, 0, 0,
        0, 1.0f / t, 0, 0,
        0, 0, 2 / (zFar - zNear), 0,
        0, 0, 0, 1;

    Eigen::Matrix4f orthogonal = orthogonal_scale * orthogonal_move;

    Eigen::Matrix4f revert_z;
    revert_z << 1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, -1, 0,
        0, 0, 0, 1;

    return revert_z * orthogonal * perspective;
}

Eigen::Matrix4f get_rotation(Vector3f axis, float angle)
{
    auto theta = deg2rad(angle);
    auto i = Eigen::Matrix3f::Identity();
    Eigen::Matrix3f a;
    a << 0, -axis.z(), axis.y(),
        axis.z(), 0, -axis.x(),
        -axis.y(), axis.x(), 0;
    Eigen::Matrix3f r = cos(theta) * i + (1 - cos(theta)) * axis * axis.transpose() + sin(theta) * a;
    Eigen::Matrix4f result;
    result << r(0, 0), r(0, 1), r(0, 2), 0,
        r(1, 0), r(1, 1), r(1, 2), 0,
        r(2, 0), r(2, 1), r(2, 2), 0,
        0, 0, 0, 1;
    return result;
}

int main(int argc, const char **argv)
{
    float angle = 0;
    bool command_line = false;
    std::string filename = "output.png";

    if (argc >= 3)
    {
        command_line = true;
        angle = std::stof(argv[2]); // -r by default
        if (argc == 4)
        {
            filename = std::string(argv[3]);
        }
        else
            return 0;
    }

    rst::rasterizer r(700, 700);

    Eigen::Vector3f eye_pos = {0, 0, 5};

    std::vector<Eigen::Vector3f> pos{{2, 0, -2}, {0, 2, -2}, {-2, 0, -2}};

    std::vector<Eigen::Vector3i> ind{{0, 1, 2}};

    auto pos_id = r.load_positions(pos);
    auto ind_id = r.load_indices(ind);

    int key = 0;
    int frame_count = 0;

    if (command_line)
    {
        r.clear(rst::Buffers::Color | rst::Buffers::Depth);

        // r.set_model(get_model_matrix(angle));
        r.set_model(get_rotation(Eigen::Vector3f{1, 0, 0}, angle));
        r.set_view(get_view_matrix(eye_pos));
        r.set_projection(get_projection_matrix(45, 1, 0.1, 50));

        r.draw(pos_id, ind_id, rst::Primitive::Triangle);
        cv::Mat image(700, 700, CV_32FC3, r.frame_buffer().data());
        image.convertTo(image, CV_8UC3, 1.0f);

        cv::imwrite(filename, image);

        return 0;
    }

    while (key != 27)
    {
        r.clear(rst::Buffers::Color | rst::Buffers::Depth);

        // r.set_model(get_model_matrix(angle));
        r.set_model(get_rotation(Eigen::Vector3f{0, 1, 0}, angle));
        r.set_view(get_view_matrix(eye_pos));
        r.set_projection(get_projection_matrix(45, 1, 0.1, 50));

        r.draw(pos_id, ind_id, rst::Primitive::Triangle);

        cv::Mat image(700, 700, CV_32FC3, r.frame_buffer().data());
        image.convertTo(image, CV_8UC3, 1.0f);
        cv::imshow("image", image);
        key = cv::waitKey(10);

        std::cout << "frame count: " << frame_count++ << '\n';

        if (key == 'a')
        {
            angle += 10;
        }
        else if (key == 'd')
        {
            angle -= 10;
        }
    }

    return 0;
}
