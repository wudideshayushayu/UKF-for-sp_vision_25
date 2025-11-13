#ifndef TOOLS__UKF
#define TOOLS__UKF

#include <Eigen/Dense>
#include <deque>
#include <functional>
#include <map>

namespace tools
{
class UKF
{
public:

    Eigen::VectorXd x;
    Eigen::MatrixXd P;

    UKF() = default;
    UKF(const double a, const double b, const double k, 
        const Eigen::VectorXd & x0, const Eigen::MatrixXd & p0,
        std::function<Eigen::VectorXd(const Eigen::VectorXd &, const Eigen::VectorXd &)> x_add = 
            [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) {return a + b; });

    Eigen::VectorXd predict(const Eigen::MatrixXd & Q, 
                            std::function<Eigen::VectorXd(const Eigen::VectorXd &)> f);
    Eigen::VectorXd update(const Eigen::VectorXd & z, const Eigen::MatrixXd & R, 
                            std::function<Eigen::VectorXd(const Eigen::VectorXd &)> h,
                            std::function<Eigen::VectorXd(const Eigen::VectorXd &, const Eigen::VectorXd &)> z_subtract =
                            [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) {return a - b;});
    

    bool diverged();

    std::map<std::string, double> data;  //卡方检验数据
    std::deque<int> recent_nis_failures{0};
    size_t window_size = 100;
    double last_nis;

private:

    Eigen::MatrixXd generate_sigma_points(const Eigen::VectorXd & x);
    Eigen::VectorXd transform(Eigen::MatrixXd & sigma_points, const Eigen::MatrixXd & sigma_points_last, std::function<Eigen::VectorXd(const Eigen::VectorXd &)> f);
    Eigen::MatrixXd P_calculate(const Eigen::VectorXd & x, const Eigen::MatrixXd & sigma_points);
    Eigen::MatrixXd P_calculate(const Eigen::VectorXd & x, const Eigen::VectorXd & z);

    Eigen::MatrixXd sigma_points_;
    Eigen::MatrixXd f_sigma_points_;
    Eigen::MatrixXd h_sigma_points_;
    Eigen::MatrixXd I;
    double a_;
    double b_;
    double k_;
    double wm0_;
    double wc0_;
    double wm_, wc_;
    double lambda_;
    std::function<Eigen::VectorXd(const Eigen::VectorXd &, const Eigen::VectorXd &)> x_add;
    
    bool diverged_ = false;
    int nees_count_ = 0;
    int nis_count_ = 0;
    int total_count_ = 0;
    bool combined_ = false;
};
}


#endif