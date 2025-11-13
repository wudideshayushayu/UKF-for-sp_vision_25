#include "UKF.hpp"

#include <numeric>

#include "tools/logger.hpp"

namespace tools
{
UKF::UKF(const double a, const double b, const double k, const Eigen::VectorXd & x0, const Eigen::MatrixXd & p0,
    std::function<Eigen::VectorXd(const Eigen::VectorXd &, const Eigen::VectorXd &)> x_add)
:a_(a), b_(b), k_(k), x(x0), P(p0), x_add(x_add)
{
    data["residual_yaw"] = 0.0;
    data["residual_pitch"] = 0.0;
    data["residual_distance"] = 0.0;
    data["residual_angle"] = 0.0;
    data["nis"] = 0.0;
    data["nees"] = 0.0;
    data["nis_fail"] = 0.0;
    data["nees_fail"] = 0.0;
    data["recent_nis_failures"] = 0.0;
    auto L = x.size();
    sigma_points_ = Eigen::MatrixXd(L, L*2 + 1);
    f_sigma_points_ = Eigen::MatrixXd(L, L*2 + 1);
    lambda_ = a_ * a_ * (L + k) - L;
    wc0_ = lambda_ / (L + lambda_) + (1 - a_ * a_ + b_);
    wm0_ = lambda_ / (L + lambda_);
    wm_ = 0.5 / (L + lambda_);
    wc_ = wm_;

}

bool UKF::diverged()
{
    return diverged_;
}

Eigen::VectorXd UKF::transform(Eigen::MatrixXd & sigma_points, const Eigen::MatrixXd & sigma_points_last, std::function<Eigen::VectorXd(const Eigen::VectorXd &)> f)
{
    Eigen::VectorXd x_mean = Eigen::VectorXd::Zero(sigma_points.rows());

    for (int i = 0; i < sigma_points_last.cols(); i++){
        
        sigma_points.col(i) = f(sigma_points_last.col(i));
        if (i > 0){
            x_mean += wm_ * sigma_points.col(i);
        }else{
            x_mean += wm0_ * sigma_points.col(i);
        }
    }

    return x_mean;
}

Eigen::MatrixXd UKF::P_calculate(const Eigen::VectorXd & x, const Eigen::VectorXd & z)
{

    Eigen::MatrixXd P_temp = Eigen::MatrixXd::Zero(x.size(), z.size());
    for (int i = 0; i < sigma_points_.cols(); i++){
        if (i == 0){
            P_temp += (f_sigma_points_.col(i) - x) * (h_sigma_points_.col(i) - z).transpose() * wc0_;
        }else{
            P_temp += (f_sigma_points_.col(i) - x) * (h_sigma_points_.col(i) - z).transpose() * wc_;
        }
    }

    return P_temp;
}

Eigen::MatrixXd UKF::P_calculate(const Eigen::VectorXd & x, const Eigen::MatrixXd & sigma_points)
{

    Eigen::MatrixXd P_temp = Eigen::MatrixXd::Zero(sigma_points.rows(), sigma_points.rows());
    for (int i = 0; i < sigma_points.cols(); i++){
        if (i == 0){
            P_temp += (sigma_points.col(i) - x) * (sigma_points.col(i) - x).transpose() * wc0_;
        }else{
            P_temp += (sigma_points.col(i) - x) * (sigma_points.col(i) - x).transpose() * wc_;
        }
    }

    return P_temp;
}

Eigen::MatrixXd UKF::generate_sigma_points(const Eigen::VectorXd & x)
{
    int n = x.size();
    double scale_factor = std::sqrt(lambda_ + n);
    auto llt0fp = P.llt();

    if(llt0fp.info() != Eigen::Success){
        tools::logger()->error("cholesky error");
        diverged_ = true;
        return Eigen::MatrixXd(); 
    }

    Eigen::MatrixXd L = llt0fp.matrixL();

    sigma_points_.col(0) = x;


    for (int i = 0; i < n; i++){
        sigma_points_.col(i + 1) = x + scale_factor * L.col(i);
        sigma_points_.col(i + n + 1) = x - scale_factor * L.col(i);
    }

    return sigma_points_;
}

Eigen::VectorXd UKF::predict(const Eigen::MatrixXd & Q,
                            std::function<Eigen::VectorXd(const Eigen::VectorXd &)> f)
{

    generate_sigma_points(x);
    x = transform(f_sigma_points_, sigma_points_, f);
    P = P_calculate(x, f_sigma_points_) + Q;

    return x;
}

Eigen::VectorXd UKF::update(const Eigen::VectorXd & z, const Eigen::MatrixXd & R,
                        std::function<Eigen::VectorXd(const Eigen::VectorXd &)> h,
                        std::function<Eigen::VectorXd(const Eigen::VectorXd &, const Eigen::VectorXd &)> z_subtract)
{

    auto x_prior = x;
    h_sigma_points_ = Eigen::MatrixXd::Zero(z.size(), x.size() * 2 + 1);
    Eigen::VectorXd z_mean = transform(h_sigma_points_, f_sigma_points_, h);
    auto S = P_calculate(z_mean, h_sigma_points_) + R;
    auto S_llt = S.llt();

    if (S_llt.info() != Eigen::Success){
        diverged_ = true;
        return x;
    }

    
    auto P_xz = P_calculate(x, z_mean);
    auto K = S_llt.solve(P_xz.transpose()).transpose();
    P = P - K * P_xz.transpose();
    x = x_add(x, K * z_subtract(z, z_mean));

    Eigen::VectorXd residual = z_subtract(z, z_mean);

    double nis = residual.transpose() * S_llt.solve(residual);
    double nees = (x - x_prior).transpose() * P.inverse() * (x - x_prior);


    constexpr double nis_threshold = 0.711;
    constexpr double nees_threshold = 0.711;

    if (nis > nis_threshold) nis_count_++, data["nis_fail"] = 1;
    if (nees > nees_threshold) nees_count_++, data["nees_fail"] = 1;
    total_count_++;
    last_nis = nis;

    recent_nis_failures.push_back(nis > nis_threshold ? 1 : 0);

    if (recent_nis_failures.size() > window_size) {
        recent_nis_failures.pop_front();
    }

    int recent_failures = std::accumulate(recent_nis_failures.begin(), recent_nis_failures.end(), 0);
    double recent_rate = static_cast<double>(recent_failures) / recent_nis_failures.size();

    data["residual_yaw"] = residual[0];
    data["residual_pitch"] = residual[1];
    data["residual_distance"] = residual[2];
    data["residual_angle"] = residual[3];
    data["nis"] = nis;
    data["nees"] = nees;
    data["recent_nis_failures"] = recent_rate;


    return x;
}

} // namespace tools
