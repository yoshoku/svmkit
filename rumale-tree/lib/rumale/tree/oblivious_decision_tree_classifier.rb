# frozen_string_literal: true

require 'numo/narray'
require 'rumale/base/classifier'

module Rumale
  module Tree
    class BaseObliviousDecisionTree < ::Rumale::Base::Estimator # rubocop:disable Style/Documentation
      class Rule # rubocop:disable Style/Documentation
        attr_reader :id, :threshold, :depth

        def initialize(id:, threshold:, depth:)
          @id = id
          @threshold = threshold
          @depth = depth
        end
      end

      class Leaf # rubocop:disable Style/Documentation
        attr_reader :label, :probs

        def initialize(label:, probs:)
          @label = label
          @probs = probs
        end
      end

      def rand_ids
        @feature_ids.sample(@params[:max_features], random: @sub_rng)
      end
    end

    class ObliviousDecisionTreeClassifier < BaseObliviousDecisionTree # rubocop:disable Metrics/ClassLength, Style/Documentation
      include ::Rumale::Base::Classifier

      attr_reader :classes, :feature_importances, :rng

      def initialize(max_depth:, min_samples_leaf: 1, max_bins: 32, criterion: 'gini', max_features: nil, random_seed: nil)
        super()
        @params = {
          criterion: criterion,
          max_depth: max_depth || 10,
          min_samples_leaf: min_samples_leaf,
          max_features: max_features,
          random_seed: random_seed || srand
        }
        @rng = Random.new(@params[:random_seed])
        @max_bins = max_bins
        @rules = []
        @leaves = []
        @depth = 0
      end

      def predict(x)
        n_samples = x.shape[0]

        predicted = Array.new(n_samples) do |i|
          leaf_id = 0
          @rules.each do |r|
            leaf_id |= (x[i, r.id] <= r.threshold ? 0 : 1) << (@depth - (r.depth + 1))
          end
          @leaves[leaf_id].label
        end

        Numo::Int32.asarray(predicted)
      end

      def predict_proba(x)
        n_samples = x.shape[0]

        predicted = Array.new(n_samples) do |i|
          leaf_id = 0
          @rules.each do |r|
            leaf_id |= (x[i, r.id] <= r.threshold ? 0 : 1) << (@depth - (r.depth + 1))
          end
          @leaves[leaf_id].probs
        end

        Numo::DFloat.asarray(predicted)
      end

      def fit(x, y) # rubocop:disable Metrics/AbcSize, Metrics/MethodLength
        n_samples = x.shape[0]
        n_features = x.shape[1]

        @params[:max_features] = n_features if @params[:max_features].nil?
        @params[:max_features] = [@params[:max_features], n_features].min
        @feature_ids = Array.new(n_features) { |v| v }
        @sub_rng = @rng.dup

        @classes = Numo::Int32.asarray(y.to_a.uniq.sort)
        candidates = Array.new(n_features) { |i| quantize_feature(x[true, i]) }

        @rules = []
        subsets = [Numo::Int32.new(n_samples).seq]

        @depth = 0
        @params[:max_depth].times do |d|
          max_gain = Float::MIN
          best_threshold = 0
          best_id = 0

          target_subsets = subsets.select { |s| s.size > @params[:min_samples_leaf] }

          rand_ids.each do |i|
            f_max_gain = Float::MIN
            f_best_threshold = 0
            candidates[i].each do |threshold|
              gain = target_subsets.sum do |target_ids|
                target_ids.size.fdiv(n_samples) * gain(threshold, x[target_ids, i], y[target_ids])
              end
              if f_max_gain < gain
                f_max_gain = gain
                f_best_threshold = threshold
              end
            end

            next unless max_gain < f_max_gain

            max_gain = f_max_gain
            best_threshold = f_best_threshold
            best_id = i
          end

          new_subsets = []
          n_non_divides = 0
          subsets.each do |target_ids|
            l_ids = x[target_ids, best_id].le(best_threshold).where.to_a
            r_ids = x[target_ids, best_id].gt(best_threshold).where.to_a
            if l_ids.size < @params[:min_samples_leaf] || r_ids.size < @params[:min_samples_leaf]
              new_subsets.push(target_ids)
              new_subsets.push(target_ids)
              n_non_divides += 1
            else
              new_subsets.push(target_ids[l_ids])
              new_subsets.push(target_ids[r_ids])
            end
          end

          # early stop if no change
          break if n_non_divides == subsets.size

          @depth = d + 1
          @rules.push(Rule.new(id: best_id, threshold: best_threshold, depth: d))

          subsets = new_subsets

          # puts "depth: #{d}, n_leaves: #{subsets.size}"
        end

        @leaves = []
        subsets.each do |target_ids|
          sz_subset = target_ids.size
          target_y = y[target_ids]
          probs = Numo::DFloat[*(@classes.to_a.map { |c| target_y.eq(c).count })] / sz_subset
          label = @classes[probs.max_index]
          @leaves.push(Leaf.new(label: label, probs: probs))
        end

        eval_importances(n_features)

        self
      end

      private

      def gain(threshold, features, labels)
        l_ids = features.le(threshold).where.to_a
        r_ids = features.gt(threshold).where.to_a

        l_impurity = 0.0
        unless l_ids.empty?
          l_labels = labels[l_ids]
          l_prob = l_labels.size.fdiv(labels.size)
          l_impurity = l_prob * impurity(l_labels)
        end

        r_impurity = 0.0
        unless r_ids.empty?
          r_labels = labels[r_ids]
          r_prob = r_labels.size.fdiv(labels.size)
          r_impurity = r_prob * impurity(r_labels)
        end

        impurity(labels) - l_impurity - r_impurity
      end

      def impurity(labels)
        # gini coefficient
        posterior_probs = Numo::DFloat[*(labels.to_a.uniq.sort.map { |c| labels.eq(c).count })] / labels.size.to_f
        1.0 - (posterior_probs**2).sum
      end

      def quantize_feature(features)
        f_max = features.max
        f_min = features.min
        f_ratio = (@max_bins - 1).fdiv(f_max - f_min)
        f_ids = Numo::Int32.cast((features - f_min) * f_ratio)
        f_ids.to_a.uniq.sort.map { |id| features[f_ids.eq(id).where].mean }
      end

      def eval_importances(n_features)
        @feature_importances = Numo::DFloat.zeros(n_features)
        @rules.each { |r| @feature_importances[r.id] += 1 }
        normalizer = @feature_importances.sum
        @feature_importances /= normalizer if normalizer.positive?
        nil
      end

      def rand_ids
        @feature_ids.sample(@params[:max_features], random: @sub_rng)
      end
    end
  end
end
