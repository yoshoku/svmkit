# Rumale

![Rumale](https://dl.dropboxusercontent.com/s/joxruk2720ur66o/rumale_header_400.png)

[![Gem Version](https://badge.fury.io/rb/rumale.svg)](https://badge.fury.io/rb/rumale)
[![BSD 3-Clause License](https://img.shields.io/badge/License-BSD%203--Clause-orange.svg)](https://github.com/yoshoku/rumale/blob/main/LICENSE.txt)
[![Documentation](https://img.shields.io/badge/api-reference-blue.svg)](https://yoshoku.github.io/rumale/doc/)

Rumale (**Ru**by **ma**chine **le**arning) is a machine learning library in Ruby.
Rumale provides machine learning algorithms with interfaces similar to Scikit-Learn in Python.
Rumale supports Support Vector Machine,
Logistic Regression, Ridge, Lasso,
Multi-layer Perceptron,
Naive Bayes, Decision Tree, Gradient Tree Boosting, Random Forest,
K-Means, Gaussian Mixture Model, DBSCAN, Spectral Clustering,
Mutidimensional Scaling, t-SNE,
Fisher Discriminant Analysis, Neighbourhood Component Analysis,
Principal Component Analysis, Non-negative Matrix Factorization,
and many other algorithms.

## Installation

Add this line to your application's Gemfile:

```ruby
gem 'rumale'
```

And then execute:

    $ bundle

Or install it yourself as:

    $ gem install rumale

## Documentation

- [Rumale API Documentation](https://yoshoku.github.io/rumale/doc/)

## Usage

### Example 1. Pendigits dataset classification

Rumale provides function loading libsvm format dataset file.
We start by downloading the pendigits dataset from LIBSVM Data web site.

```bash
$ wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/pendigits
$ wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/pendigits.t
```

Training of the classifier with Linear SVM and RBF kernel feature map is the following code.

```ruby
require 'rumale'

# Load the training dataset.
samples, labels = Rumale::Dataset.load_libsvm_file('pendigits')

# Map training data to RBF kernel feature space.
transformer = Rumale::KernelApproximation::RBF.new(gamma: 0.0001, n_components: 1024, random_seed: 1)
transformed = transformer.fit_transform(samples)

# Train linear SVM classifier.
classifier = Rumale::LinearModel::SVC.new(reg_param: 0.0001, random_seed: 1)
classifier.fit(transformed, labels)

# Save the model.
File.open('transformer.dat', 'wb') { |f| f.write(Marshal.dump(transformer)) }
File.open('classifier.dat', 'wb') { |f| f.write(Marshal.dump(classifier)) }
```

Classifying testing data with the trained classifier is the following code.

```ruby
require 'rumale'

# Load the testing dataset.
samples, labels = Rumale::Dataset.load_libsvm_file('pendigits.t')

# Load the model.
transformer = Marshal.load(File.binread('transformer.dat'))
classifier = Marshal.load(File.binread('classifier.dat'))

# Map testing data to RBF kernel feature space.
transformed = transformer.transform(samples)

# Classify the testing data and evaluate prediction results.
puts("Accuracy: %.1f%%" % (100.0 * classifier.score(transformed, labels)))

# Other evaluating approach
# results = classifier.predict(transformed)
# evaluator = Rumale::EvaluationMeasure::Accuracy.new
# puts("Accuracy: %.1f%%" % (100.0 * evaluator.score(results, labels)))
```

Execution of the above scripts result in the following.

```bash
$ ruby train.rb
$ ruby test.rb
Accuracy: 98.7%
```

### Example 2. Cross-validation

```ruby
require 'rumale'

# Load dataset.
samples, labels = Rumale::Dataset.load_libsvm_file('pendigits')

# Define the estimator to be evaluated.
lr = Rumale::LinearModel::LogisticRegression.new

# Define the evaluation measure, splitting strategy, and cross validation.
ev = Rumale::EvaluationMeasure::Accuracy.new
kf = Rumale::ModelSelection::StratifiedKFold.new(n_splits: 5, shuffle: true, random_seed: 1)
cv = Rumale::ModelSelection::CrossValidation.new(estimator: lr, splitter: kf, evaluator: ev)

# Perform 5-cross validation.
report = cv.perform(samples, labels)

# Output result.
mean_accuracy = report[:test_score].sum / kf.n_splits
puts "5-CV mean accuracy: %.1f%%" % (100.0 * mean_accuracy)
```

Execution of the above scripts result in the following.

```bash
$ ruby cross_validation.rb
5-CV mean accuracy: 95.4%
```


## Speedup

### Numo::Linalg
Rumale uses [Numo::NArray](https://github.com/ruby-numo/numo-narray) for typed arrays.
Loading the [Numo::Linalg](https://github.com/ruby-numo/numo-linalg) allows to perform matrix and vector product of Numo::NArray
using BLAS libraries.
Some machine learning algorithms frequently compute matrix and vector products,
the execution speed of such algorithms can be expected to be accelerated.

Install Numo::Linalg gem.

```bash
$ gem install numo-linalg
```

In ruby script, just load Numo::Linalg along with Rumale.

```ruby
require 'numo/linalg/autoloader'
require 'rumale'
```

### Numo::OpenBLAS
[Numo::OpenBLAS](https://github.com/yoshoku/numo-openblas) downloads and builds OpenBLAS during installation
and uses that as a background library for Numo::Linalg.

Install compilers for building OpenBLAS.

```bash
$ sudo apt-get install gcc gfortran make
```

Install Numo::OpenBLAS gem.

```bash
$ gem install numo-openblas
```

Load Numo::OpenBLAS gem instead of Numo::Linalg.

```ruby
require 'numo/openblas'
require 'rumale'
```

### Numo::BLIS
[Numo::BLIS](https://github.com/yoshoku/numo-blis) downloads and builds BLIS during installation
and uses that as a background library for Numo::Linalg.
BLIS is one of the high-performance BLAS as with OpenBLAS,
and using that can be expected to speed up of processing in Rumale.

Install Numo::BLIS gem.

```bash
$ gem install numo-blis
```

Load Numo::BLIS gem instead of Numo::Linalg.

```ruby
require 'numo/blis'
require 'rumale'
```

### Parallel
Several estimators in Rumale support parallel processing.
Parallel processing in Rumale is realized by [Parallel](https://github.com/grosser/parallel) gem,
so install and load it.

```bash
$ gem install parallel
```

```ruby
require 'parallel'
require 'rumale'
```

Estimators that support parallel processing have n_jobs parameter.
When -1 is given to n_jobs parameter, all processors are used.

```ruby
estimator = Rumale::Ensemble::RandomForestClassifier.new(n_jobs: -1, random_seed: 1)
```

## Related Projects

- [Rumale::SVM](https://github.com/yoshoku/rumale-svm) provides support vector machine algorithms in LIBSVM and LIBLINEAR with Rumale interface.
- [Rumale::Torch](https://github.com/yoshoku/rumale-torch) provides the learning and inference by the neural network defined in torch.rb with Rumale interface.

## License

The gem is available as open source under the terms of the [BSD-3-Clause License](https://opensource.org/licenses/BSD-3-Clause).
