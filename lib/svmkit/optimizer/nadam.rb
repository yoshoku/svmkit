# frozen_string_literal: true

require 'svmkit/validation'

module SVMKit
  # This module consists of the classes that implement optimizers adaptively tuning hyperparameters.
  module Optimizer
    # Nadam is a class that implements Nadam optimizer.
    # This class is used for internal processes.
    #
    # *Reference*
    # - T. Dozat, "Incorporating Nesterov Momentum into Adam," Tech. Repo. Stanford University, 2015.
    class Nadam
      include Validation

      # Create a new optimizer with Nadam
      #
      # @param learning_rate [Float] The initial value of learning rate.
      # @param momentum [Float] The initial value of momentum.
      # @param decay1 [Float] The smoothing parameter for the first moment.
      # @param decay2 [Float] The smoothing parameter for the second moment.
      # @param schedule_decay [Float] The smooting parameter.
      def initialize(learning_rate: 0.01, momentum: 0.9, decay1: 0.9, decay2: 0.999)
        check_params_float(learning_rate: learning_rate, momentum: momentum, decay1: decay1, decay2: decay2)
        check_params_positive(learning_rate: learning_rate, momentum: momentum, decay1: decay1, decay2: decay2)
        @params = {}
        @params[:learning_rate] = learning_rate
        @params[:momentum] = momentum
        @params[:decay1] = decay1
        @params[:decay2] = decay2
        @fst_moment = nil
        @sec_moment = nil
        @decay1_prod = 1.0
        @iter = 0
      end

      # Calculate the updated weight with Nadam adaptive learning rate.
      #
      # @param weight [Numo::DFloat] (shape: [n_features]) The weight to be updated.
      # @param gradient [Numo::DFloat] (shape: [n_features]) The gradient for updating the weight.
      # @return [Numo::DFloat] (shape: [n_feautres]) The updated weight.
      def call(weight, gradient)
        @fst_moment ||= Numo::DFloat.zeros(weight.shape[0])
        @sec_moment ||= Numo::DFloat.zeros(weight.shape[0])

        @iter += 1

        decay1_curr = @params[:decay1] * (1.0 - 0.5 * 0.96**(@iter * 0.004))
        decay1_next = @params[:decay1] * (1.0 - 0.5 * 0.96**((@iter + 1) * 0.004))
        decay1_prod_curr = @decay1_prod * decay1_curr
        decay1_prod_next = @decay1_prod * decay1_curr * decay1_next
        @decay1_prod = decay1_prod_curr

        @fst_moment = @params[:decay1] * @fst_moment + (1.0 - @params[:decay1]) * gradient
        @sec_moment = @params[:decay2] * @sec_moment + (1.0 - @params[:decay2]) * gradient**2
        nm_gradient = gradient / (1.0 - decay1_prod_curr)
        nm_fst_moment = @fst_moment / (1.0 - decay1_prod_next)
        nm_sec_moment = @sec_moment / (1.0 - @params[:decay2]**@iter)

        weight - (@params[:learning_rate] / (nm_sec_moment**0.5 + 1e-8)) * ((1 - decay1_curr) * nm_gradient + decay1_next * nm_fst_moment)
      end
    end
  end
end
