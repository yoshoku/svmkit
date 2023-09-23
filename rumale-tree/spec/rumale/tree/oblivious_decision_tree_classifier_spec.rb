# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::Tree::ObliviousDecisionTreeClassifier do
  let(:x) { dataset[0] }
  let(:y) { dataset[1] }
  let(:classes) { y.to_a.uniq.sort }
  let(:n_samples) { x.shape[0] }
  let(:n_features) { x.shape[1] }
  let(:n_classes) { classes.size }
  let(:criterion) { 'gini' }
  let(:max_depth) { nil }
  let(:min_samples_leaf) { 1 }
  let(:max_features) { nil }
  let(:estimator) do
    described_class.new(criterion: criterion, max_depth: max_depth,
                        min_samples_leaf: min_samples_leaf, max_features: max_features, random_seed: 1).fit(x, y)
  end
  let(:probs) { estimator.predict_proba(x) }
  let(:predicted_by_probs) { Numo::Int32[*(Array.new(n_samples) { |n| classes[probs[n, true].max_index] })] }
  let(:score) { estimator.score(x, y) }

  context 'when binary classification problem' do
    let(:dataset) { two_clusters_dataset }

    it 'classifies two clusters data', :aggregate_failures do
      # expect(estimator.tree).to be_a(Rumale::Tree::Node)
      expect(estimator.classes).to be_a(Numo::Int32)
      expect(estimator.classes).to be_contiguous
      expect(estimator.classes.ndim).to eq(1)
      expect(estimator.classes.shape[0]).to eq(n_classes)
      expect(estimator.feature_importances).to be_a(Numo::DFloat)
      expect(estimator.feature_importances).to be_contiguous
      expect(estimator.feature_importances.ndim).to eq(1)
      expect(estimator.feature_importances.shape[0]).to eq(n_features)
      expect(score).to eq(1.0)
    end
  end

  context 'when multiclass classification problem' do
    let(:dataset) { three_clusters_dataset }

    it 'classifies three clusters data', :aggregate_failures do
      # expect(estimator.tree).to be_a(Rumale::Tree::Node)
      expect(estimator.classes).to be_a(Numo::Int32)
      expect(estimator.classes).to be_contiguous
      expect(estimator.classes.ndim).to eq(1)
      expect(estimator.classes.shape[0]).to eq(n_classes)
      expect(estimator.feature_importances).to be_a(Numo::DFloat)
      expect(estimator.feature_importances).to be_contiguous
      expect(estimator.feature_importances.ndim).to eq(1)
      expect(estimator.feature_importances.shape[0]).to eq(n_features)
      expect(score).to eq(1.0)
    end

    it 'estimates class probabilities with three clusters dataset', :aggregate_failures do
      expect(probs).to be_a(Numo::DFloat)
      expect(probs).to be_contiguous
      expect(probs.ndim).to eq(2)
      expect(probs.shape[0]).to eq(n_samples)
      expect(probs.shape[1]).to eq(n_classes)
      expect(predicted_by_probs).to eq(y)
    end

    context 'when max_depth parameter is given' do
      let(:max_depth) { 1 }

      it 'learns model with given parameters', :aggregate_failures do
        expect(estimator.params[:max_depth]).to eq(max_depth)
        # expect(estimator.tree.left.left).to be_nil
        # expect(estimator.tree.left.right).to be_nil
        # expect(estimator.tree.right.left).to be_nil
        # expect(estimator.tree.right.right).to be_nil
      end
    end

    context 'when min_samples_leaf parameter is given' do
      let(:min_samples_leaf) { 110 }

      it 'learns model with given parameters', :aggregate_failures do
        expect(estimator.params[:min_samples_leaf]).to eq(min_samples_leaf)
        # expect(estimator.tree.left.leaf).to be_truthy
        # expect(estimator.tree.left.n_samples).to be >= min_samples_leaf
        # expect(estimator.tree.right).to be_nil
      end
    end

    context 'when max_features parameter is given' do
      context 'with negative value' do
        let(:max_features) { -10 }

        xit 'raises ArgumentError by validation' do # rubocop:disable RSpec/PendingWithoutReason
          expect { estimator }.to raise_error(ArgumentError)
        end
      end

      context 'with value larger than number of features' do
        let(:max_features) { 10 }

        xit 'value of max_features is equal to the number of features' do # rubocop:disable RSpec/PendingWithoutReason
          expect(estimator.params[:max_features]).to eq(x.shape[1])
        end
      end

      context 'with valid value' do
        let(:max_features) { 2 }

        it 'learns model with given parameters' do
          expect(estimator.params[:max_features]).to eq(2)
        end
      end
    end
  end
end
