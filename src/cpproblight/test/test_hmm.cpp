#include <cpproblight.h>
#include "xtensor/xadapt.hpp"

// Hidden Markov model
// http://www.robots.ox.ac.uk/~fwood/assets/pdf/Wood-AISTATS-2014.pdf

xt::xarray<double> forward(xt::xarray<double> observation)
{
  auto init_dist = cpproblight::distributions::Categorical(xt::xarray<double> {1, 1, 1});
  cpproblight::distributions::Categorical trans_dists[] = {
      cpproblight::distributions::Categorical(xt::xarray<double> {0.1, 0.5, 0.4}),
      cpproblight::distributions::Categorical(xt::xarray<double> {0.2, 0.2, 0.6}),
      cpproblight::distributions::Categorical(xt::xarray<double> {0.15, 0.15, 0.7})
  };
  cpproblight::distributions::Normal obs_dists[] = {
      cpproblight::distributions::Normal(-1, 1),
      cpproblight::distributions::Normal(1, 1),
      cpproblight::distributions::Normal(0, 1)
  };
  int init_state = cpproblight::sample(init_dist)(0);
  std::vector<int> states {init_state};

  for (auto & o : observation)
  {
    int state = cpproblight::sample(trans_dists[states.back()])(0);
    cpproblight::observe(obs_dists[states.back()], o);
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
  cpproblight::Model model = cpproblight::Model(forward, xt::xarray<double> {}, "Hidden Markov model C++");
  model.startServer(serverAddress);
  return 0;
}
