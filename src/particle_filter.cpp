/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
    static std::default_random_engine gen;
    normal_distribution<double> dist_x(0.0, std[0]);
    normal_distribution<double> dist_y(0.0, std[1]);
    normal_distribution<double> dist_theta(0.0, std[2]);

    num_particles = 100;
    Particle particle_temp;
    for (int i = 0; i < num_particles; i++){
        particle_temp.id = i;
        particle_temp.x = x + dist_x(gen);
        particle_temp.y = y + dist_y(gen);
        particle_temp.theta = theta + dist_theta(gen);
        particle_temp.weight = 1.0;
        particles.push_back(particle_temp);
        weights.push_back(1.0);
    }
    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
    static std::default_random_engine gen;
    normal_distribution<double> dist_x(0.0, std_pos[0]);
    normal_distribution<double> dist_y(0.0, std_pos[1]);
    normal_distribution<double> dist_theta(0.0, std_pos[2]);
    for (int i = 0; i < num_particles; i++){
        if (fabs(yaw_rate) < 1e-3){
            particles[i].x += velocity*delta_t*cos(particles[i].theta);
            particles[i].y += velocity*delta_t*sin(particles[i].theta);
        }
        else{
            particles[i].x += (velocity/yaw_rate)*(sin(particles[i].theta + yaw_rate*delta_t) - sin(particles[i].theta));
            particles[i].y += (velocity/yaw_rate)*(cos(particles[i].theta) - cos(particles[i].theta + yaw_rate*delta_t));
            particles[i].theta += yaw_rate*delta_t;
        }

        particles[i].x += dist_x(gen);
        particles[i].y += dist_y(gen);
        particles[i].theta += dist_theta(gen);
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
    for (int o = 0; o < observations.size(); o++){
        int matched_id = 0;
        double min_dist_square = INFINITY;
        for (int l = 0; l < predicted.size(); l++){
            double dist_square = (predicted[l].x - observations[o].x)*(predicted[l].x - observations[o].x) + (predicted[l].y - observations[o].y)*(predicted[l].y - observations[o].y);
            if (dist_square < min_dist_square){
                matched_id = predicted[l].id;
                min_dist_square = dist_square;
            }
        }
        observations[o].id = matched_id;
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
    double range_square = sensor_range * sensor_range;
    for (int i = 0; i < num_particles; i++){
        // get all landmarks in the sensor range
        std::vector<LandmarkObs> predictions;
        for (int l = 0; l < map_landmarks.landmark_list.size(); l++){
            double delta_x = map_landmarks.landmark_list[l].x_f - particles[i].x;
            double delta_y = map_landmarks.landmark_list[l].y_f - particles[i].y;
            if (delta_x*delta_x + delta_y*delta_y < range_square){
                LandmarkObs landmark_temp;
                landmark_temp.x = map_landmarks.landmark_list[l].x_f;
                landmark_temp.y = map_landmarks.landmark_list[l].y_f;
                landmark_temp.id = map_landmarks.landmark_list[l].id_i;
                predictions.push_back(landmark_temp);
            }
        }

        if (predictions.size() == 0){
            particles[i].weight = 0.0;
            weights[i] = 0.0;
            continue; // ignore the below code, run the next loop;
        }

        //transform the measured landmarks from vehicle's coordinate into map's coordinate
        std::vector<LandmarkObs> transformed_obs;
        for (int o = 0; o < observations.size(); o++){
            LandmarkObs obs;
            obs.id = observations[o].id;
            obs.x = particles[i].x + observations[o].x * cos(particles[i].theta) - observations[o].y * sin(particles[i].theta);
            obs.y = particles[i].y + observations[o].x * sin(particles[i].theta) + observations[o].y * cos(particles[i].theta);
            transformed_obs.push_back(obs);
        }

        //associate the measured landmarks to ones in map;
        dataAssociation(predictions, transformed_obs);

        //update weights
        double total_weights = 1.0;
        for (int o = 0; o < transformed_obs.size(); o++){
            int index = transformed_obs[o].id - 1;
            double delta_x = map_landmarks.landmark_list[index].x_f - transformed_obs[o].x;
            double delta_y = map_landmarks.landmark_list[index].y_f - transformed_obs[o].y;
            double gauss_norm = 1/(2 * M_PI * std_landmark[0] * std_landmark[1]);
            double exponent = (delta_x * delta_x)/(2.0 * std_landmark[0] * std_landmark[0]) + (delta_y * delta_y)/(2.0 * std_landmark[1] * std_landmark[1]);
            total_weights *= gauss_norm * exp(-exponent);
        }
        particles[i].weight = total_weights;
        weights[i] = total_weights;
    }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
    static std::default_random_engine gen;
    std::discrete_distribution<int> dist_weight(weights.begin(), weights.end());
    std::vector<Particle> new_particles;
    new_particles.reserve((unsigned long)num_particles);

    for (int i = 0; i < num_particles; i++){
        int index = dist_weight(gen);
        new_particles.push_back(particles[index]);
    }
    particles = std::move(new_particles);

#ifdef DEBUG_OUTPUT
    cout << "---------------------------resampling------------------------------" << endl;
    for (int i = 0; i < particles.size(); i++){
        cout << "particle: " << i << "\t" << particles[i].x << "\t" << particles[i].y << "\t" << particles[i].theta << "\t" << particles[i].weight << endl;
    }
#endif
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
