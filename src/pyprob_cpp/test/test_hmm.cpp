#include <pyprob_cpp.h>
#include "xtensor/xadapt.hpp"

// Hidden Markov model
// http://www.robots.ox.ac.uk/~fwood/assets/pdf/Wood-AISTATS-2014.pdf

xt::xarray<double> forward()
{
  auto init_dist = pyprob_cpp::distributions::Categorical(xt::xarray<double> {1, 1, 1});
  pyprob_cpp::distributions::Categorical trans_dists[] = {
      pyprob_cpp::distributions::Categorical(xt::xarray<double> {0.1, 0.5, 0.4}),
      pyprob_cpp::distributions::Categorical(xt::xarray<double> {0.2, 0.2, 0.6}),
      pyprob_cpp::distributions::Categorical(xt::xarray<double> {0.15, 0.15, 0.7})
  };
  pyprob_cpp::distributions::Normal obs_dists[] = {
      pyprob_cpp::distributions::Normal(-1, 1),
      pyprob_cpp::distributions::Normal(1, 1),
      pyprob_cpp::distributions::Normal(0, 1)
  };
  int init_state = pyprob_cpp::sample(init_dist)(0);
  std::vector<int> states {init_state};

  for (int i = 0; i < 16; i++)
  {
    int state = pyprob_cpp::sample(trans_dists[states.back()])(0);
    pyprob_cpp::observe(obs_dists[state], "obs" + std::to_string(i));
    states.push_back(state);
  }

  std::vector<double> v;
  for (int i = 0; i < states.size(); i++)
  {
    for (int j = 0; j < 3; j++)
    {
      v.push_back(j == states[i] ? 1.0 : 0.0);
    }
  }
  return xt::adapt(v, std::vector<int32_t> {int(states.size()), 3});
}


int main(int argc, char *argv[])
{
  auto serverAddress = (argc > 1) ? argv[1] : "tcp://*:5555";
  pyprob_cpp::Model model = pyprob_cpp::Model(forward, "Hidden Markov model C++");
  model.startServer(serverAddress);
  return 0;
}
