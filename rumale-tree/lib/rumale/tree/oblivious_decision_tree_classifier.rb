# frozen_string_literal: true

require 'numo/narray'
require 'rumale/base/classifier'

module Rumale
  module Tree
    class BaseObliviousDecisionTree < ::Rumale::Base::Estimator # rubocop:disable Style/Documentation
      include ::Rumale::Tree::ExtBaseObliviousDecisionTree

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

      class Subset # rubocop:disable Style/Documentation
        attr_reader :indices, :leaf

        def initialize(indices: nil, leaf: false)
          @indices = indices
          @leaf = leaf
        end
      end

      def rand_ids
        @feature_ids.sample(@params[:max_features], random: @sub_rng)
      end
    end

    class ObliviousDecisionTreeClassifier < BaseObliviousDecisionTree # rubocop:disable Style/Documentation
      include ::Rumale::Base::Classifier
      include ::Rumale::Tree::ExtObliviousDecisionTreeClassifier

      attr_reader :classes, :feature_importances, :rng

      def initialize(max_depth:, min_samples_leaf: 1, max_bins: 128, criterion: 'gini', max_features: nil, random_seed: nil)
        super()
        @params = {
          criterion: criterion,
          max_depth: max_depth || 100,
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

        y = Numo::Int32.cast(y) unless y.is_a?(Numo::Int32)
        uniq_y = y.to_a.uniq.sort
        @classes = Numo::Int32.asarray(uniq_y)
        y = y.map { |v| uniq_y.index(v) }
        n_classes = @classes.size

        # candidates = Array.new(n_features) { |i| quantize_feature(x[true, i]) }
        # candidates = Array.new(n_features) { |i| sort_uniq(x[true, i]) }
        candidates = Array.new(n_features) { |i| sort_uniq(quantize_feature(x[true, i])) }

        @rules = []
        subsets = [Subset.new(indices: Array.new(n_samples) { |v| v })]

        @depth = 0
        @params[:max_depth].times do |d|
          max_gain = Float::MIN
          best_threshold = 0
          best_id = 0

          target_subsets = subsets.reject(&:leaf)

          rand_ids.each do |i|
            f_best_threshold, f_max_gain = find_threshold(candidates[i], x[true, i], y, target_subsets, 'gini', n_classes).to_a

            next unless max_gain < f_max_gain

            max_gain = f_max_gain
            best_threshold = f_best_threshold
            best_id = i
          end

          new_subsets = []
          n_non_divides = 0
          subsets.each do |s|
            target_ids = s.indices
            if s.leaf
              new_subsets.push(Subset.new(indices: target_ids, leaf: true))
              new_subsets.push(Subset.new(indices: target_ids, leaf: true))
              n_non_divides += 1
            else
              l_ids = x[target_ids, best_id].le(best_threshold).where.to_a
              r_ids = x[target_ids, best_id].gt(best_threshold).where.to_a
              if l_ids.size < @params[:min_samples_leaf] || r_ids.size < @params[:min_samples_leaf]
                new_subsets.push(Subset.new(indices: target_ids, leaf: true))
                new_subsets.push(Subset.new(indices: target_ids, leaf: true))
                n_non_divides += 1
              else
                new_subsets.push(Subset.new(indices: Numo::Int32.cast(target_ids)[l_ids].to_a))
                new_subsets.push(Subset.new(indices: Numo::Int32.cast(target_ids)[r_ids].to_a))
              end
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
        subsets.each do |s|
          target_ids = s.indices
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

      def quantize_feature(features)
        f_max = features.max
        f_min = features.min
        f_ratio = (@max_bins - 1).fdiv(f_max - f_min)
        f_ids = Numo::Int32.cast((features - f_min) * f_ratio)
        Numo::DFloat.cast(f_ids.to_a.uniq.sort.map { |id| features[f_ids.eq(id).where].mean })
      end

      def eval_importances(n_features)
        @feature_importances = Numo::DFloat.zeros(n_features)
        @rules.each { |r| @feature_importances[r.id] += 1 }
        normalizer = @feature_importances.sum
        @feature_importances /= normalizer if normalizer.positive?
        nil
      end
    end
  end
end
