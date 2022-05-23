https://www.analyticsvidhya.com/blog/2018/06/comprehensive-data-science-machine-learning-interview-guide/
https://www.analyticsvidhya.com/blog/category/career/interviews/
20 Probability Interview Problems Asked By Top-Tech Companies & Wall Street

    [Facebook - Easy] There is a fair coin (one side heads, one side tails) and an unfair coin (both sides tails). You pick one at random, flip it 5 times, and observe that it comes up as tails all five times. What is the chance that you are flipping the unfair coin?
    [Lyft - Easy] You and your friend are playing a game. The two of you will continue to toss a coin until the sequence HH or TH shows up. If HH shows up first, you win. If TH shows up first, your friend wins. What is the probability of you winning?
    [Google - Easy] What is the probability that a seven-game series goes to 7 games?
    [Facebook - Easy] Facebook has a content team that labels pieces of content on the platform as spam or not spam. 90% of them are diligent raters and will label 20% of the content as spam and 80% as non-spam. The remaining 10% are non-diligent raters and will label 0% of the content as spam and 100% as non-spam. Assume the pieces of content are labeled independently from one another, for every rater. Given that a rater has labeled 4 pieces of content as good, what is the probability that they are a diligent rater?
    [Bloomberg - Easy] Say you draw a circle and choose two chords at random. What is the probability that those chords will intersect?
    [Amazon - Easy] 1/1000 people have a particular disease, and there is a test that is 98% correct if you have the disease. If you don’t have the disease, there is a 1% error rate. If someone tests positive, what are the odds they have the disease?
    [Facebook - Easy] There are 50 cards of 5 different colors. Each color has cards numbered between 1 to 10. You pick 2 cards at random. What is the probability that they are not of same color and also not of same number?
    [Tesla - Easy] A fair six-sided die is rolled twice. What is the probability of getting 1 on the first roll and not getting 6 on the second roll?
    [Facebook - Easy] What is the expected number of rolls needed to see all 6 sides of a fair die?
    [Microsoft - Easy] Three friends in Seattle each told you it’s rainy, and each person has a 1/3 probability of lying. What is the probability that Seattle is rainy? Assume the probability of rain on any given day in Seattle is 0.25.
    [Uber - Easy] Say you roll three dice, one by one. What is the probability that you obtain 3 numbers in a strictly increasing order?
    [Bloomberg - Medium] Three ants are sitting at the corners of an equilateral triangle. Each ant randomly picks a direction and starts moving along the edge of the triangle. What is the probability that none of the ants collide? Now, what if it is k ants on all k corners of an equilateral polygon?
    [Two Sigma - Medium] What is the expected number of coin flips needed to get two consecutive heads?
    [Amazon - Medium] How many cards would you expect to draw from a standard deck before seeing the first ace?
    [Robinhood - Medium] A and B are playing a game where A has n+1 coins, B has n coins, and they each flip all of their coins. What is the probability that A will have more heads than B?
    [Airbnb - Medium] Say you are given an unfair coin, with an unknown bias towards heads or tails. How can you generate fair odds using this coin?
    [Quora - Medium] Say you have N i.i.d. draws of a normal distribution with parameters μ and σ. What is the probability that k of those draws are larger than some value Y?
    [Spotify - Hard] A fair die is rolled n times. What is the probability that the largest number rolled is r, for each r in 1..6?
    [Snapchat - Hard] There are two groups of n users, A and B, and each user in A is friends with those in B and vice versa. Each user in A will randomly choose a user in B as their best friend and each user in B will randomly choose a user in A as their best friend. If two people have chosen each other, they are mutual best friends. What is the probability that there will be no mutual best friendships?
    [Tesla - Hard] Suppose there is a new vehicle launch upcoming. Initial data suggests that any given day there is either a malfunction with some part of the vehicle or possibility of a crash, with probability p which then requires a replacement. Additionally, each vehicle that has been around for n days must be replaced. What is the long-term frequency of vehicle replacements?

20 Statistics Problems Asked By FANG & Hedge Funds

    [Facebook - Easy] How would you explain a confidence interval to a non-technical audience?
    [Two Sigma - Easy] Say you are running a multiple linear regression and believe there are several predictors that are correlated. How will the results of the regression be affected if they are indeed correlated? How would you deal with this problem?
    [Uber - Easy] Describe p-values in layman’s terms.
    [Facebook - Easy] How would you build and test a metric to compare two user’s ranked lists of movie/tv show preferences?
    [Microsoft - Easy] Explain the statistical background behind power.
    [Twitter - Easy] Describe A/B testing. What are some common pitfalls?
    [Google - Medium] How would you derive a confidence interval from a series of coin tosses?
    [Stripe - Medium] Say you model the lifetime for a set of customers using an exponential distribution with parameter λ, and you have the lifetime history (in months) of n customers. What is your best guess for λ?
    [Lyft - Medium] Derive the mean and variance of the uniform distribution U(a, b).
    [Google - Medium] Say we have X ~ Uniform(0, 1) and Y ~ Uniform(0, 1). What is the expected value of the minimum of X and Y?
    [Spotify - Medium] You sample from a uniform distribution [0, d] n times. What is your best estimate of d?
    [Quora - Medium] You are drawing from a normally distributed random variable X ~ N(0, 1) once a day. What is the approximate expected number of days until you get a value of more than 2?
    [Facebook - Medium] Derive the expectation for a geometric distributed random variable.
    [Google - Medium] A coin was flipped 1000 times, and 550 times it showed up heads. Do you think the coin is biased? Why or why not?
    [Robinhood - Medium] Say you have n integers 1…n and take a random permutation. For any integers i, j let a swap be defined as when the integer i is in the jth position, and vice versa. What is the expected value of the total number of swaps?
    [Uber - Hard] What is the difference between MLE and MAP? Describe it mathematically.
    [Google - Hard] Say you have two subsets of a dataset for which you know their means and standard deviations. How do you calculate the blended mean and standard deviation of the total dataset? Can you extend it to K subsets?
    [Lyft - Hard] How do you randomly sample a point uniformly from a circle with radius 1?
    [Two Sigma - Hard] Say you continually sample from some i.i.d. uniformly distributed (0, 1) random variables until the sum of the variables exceeds 1. How many times do you expect to sample?
    [Uber - Hard] Given a random Bernoulli trial generator, how do you return a value sampled from a normal distribution

Solutions To Probability Interview Questions

Problem #1 Solution:

We can use Bayes Theorem here. Let U denote the case where we are flipping the unfair coin and F denote the case where we are flipping a fair coin. Since the coin is chosen randomly, we know that P(U) = P(F) = 0.5. Let 5T denote the event where we flip 5 heads in a row. Then we are interested in solving for P(U|5T), i.e., the probability that we are flipping the unfair coin, given that we saw 5 tails in a row.

We know P(5T|U) = 1 since by definition the unfair coin will always result in tails. Additionally, we know that P(5T|F) = 1/2^5 = 1/32 by definition of a fair coin. By Bayes Theorem we have:

Therefore the probability we picked the unfair coin is about 97%.

Problem #5 Solution:

By definition, a chord is a line segment whereby the two endpoints lie on the circle. Therefore, two arbitrary chords can always be represented by any four points chosen on the circle. If you choose to represent the first chord by two of the four points then you have:

choices of choosing the two points to represent chord 1 (and hence the other two will represent chord 2). However, note that in this counting, we are duplicating the count of each chord twice since a chord with endpoints p1 and p2 is the same as a chord with endpoints p2 and p1. Therefore the proper number of valid chords is:

Among these three configurations, only exactly one of the chords will intersect, hence the desired probability is:

Problem #13 Solution:

Let X be the number of coin flips needed until two heads. Then we want to solve for E[X]. Let H denote a flip that resulted in heads, and T denote a flip that resulted in tails. Note that E[X] can be written in terms of E[X|H] and E[X|T], i.e. the expected number of flips needed, conditioned on a flip being either heads or tails respectively.

Conditioning on the first flip, we have:

Note that E[X|T] = E[X] since if a tail is flipped, we need to start over in getting two heads in a row.

To solve for E[X|H], we can condition it further on the next outcome: either heads (HH) or tails (HT).

Therefore, we have:

Note that if the result is HH, then E[X|HH] = 0 since the outcome was achieved, and that E[X|HT] = E[X] since a tail was flipped, we need to start over again, so:

Plugging this into the original equation yields E[X] = 6 coin flips

Problem #15 Solution:

Consider the first n coins that A flips, versus the n coins that B flips.

There are three possible scenarios:

    A has more heads than B
    A and B have an equal amount of heads
    A has less heads than B

Notice that in scenario 1, A will always win (irrespective of coin n+1), and in scenario 3, A will always lose (irrespective of coin n+1). By symmetry, these two scenarios have an equal probability of occurring.

Denote the probability of either scenario as x, and the probability of scenario 2 as y.

We know that 2x + y = 1 since these 3 scenarios are the only possible outcomes. Now let’s consider coin n+1. If the flip results in heads, with probability 0.5, then A will have won after scenario 2 (which happens with probability y). Therefore, A’s total chances of winning the game are increased by 0.5y.

Thus, the probability that A will win the game is:

Problem #18 Solution:

Let B be the event that all n rolls have a value less than or equal to r. Then we have:

since all n rolls must have a value less than or equal to r. Let A be the event that the largest number is r. We have:

and since the two events on the right hand side are disjoint, we have:

Therefore, the probability of A is given by:

Solutions To Statistics Interview Questions

Problem #2 Solution:

There will be two main problems. The first is that the coefficient estimates and signs will vary dramatically, depending on what particular variables you include in the model. In particular, certain coefficients may even have confidence intervals that include 0 (meaning it is difficult to tell whether an increase in that X value is associated with an increase or decrease in Y). The second is that the resulting p-values will be misleading - an important variable might have a high p-value and deemed insignificant even though it is actually important.

You can deal with this problem by either removing or combining the correlated predictors. In removing the predictors, it is best to understand the causes of the correlation (i.e. did you include extraneous predictors or such as both X and 2X). For combining predictors, it is possible to include interaction terms (the product of the two). Lastly, you should also 1) center data, and 2) try to obtain a larger sample size (which will lead to narrower confidence intervals).

Problem #9 Solution:

For X ~U(a, b) we have the following:

Therefore we can calculate the mean as:

Similarly for variance we want:

And we have:

Therefore:

Problem #12 Solution:

Since X is normally distributed, we can look at the cumulative distribution function (CDF) of the normal distribution:

To check the probability X is at least 2, we can check (knowing that X is distributed as standard normal):

Therefore P(X > 2) = 1 - 0.977 = 0.023 for any given day. Since the draws are independent each day, then the expected time until drawing an X > 2 follows a geometric distribution, with p = 0.023. Let T be a random variable denoting the number of days, then we have:

Problem #14 Solution:

Because the sample size of flips is large (1000), we can apply the Central Limit Theorem. Since each individual flip is a Bernoulli random variable, we can assume it has a probability of showing up heads as p. Then we want to test whether p is 0.5 (i.e. whether it is fair). The Central Limit Theorem allows us to approximate the total number of heads seen as being normally distributed.

More specifically, the number of heads seen should follow a Binomial distribution since it a sum of Bernoulli random variables. If the coin is not biased (p = 0.5), then we have the following on the expected number of heads:

and the variance is given by:

Since this mean and standard deviation specify the normal distribution, we can calculate the corresponding z-score for 550 heads:

This means that, if the coin were fair, the event of seeing 550 heads should occur with a < 1% chance under normality assumptions. Therefore, the coin is likely biased.

Problem #20 Solution:

Assume we have n Bernoulli trials each with a success probability of p:

Assuming iid trials, we can compute the sample mean for p from a large number of trials:

We know the expectation of this sample mean is:

Additionally, we can compute the variance of this sample mean:

Assume we sample a large n. Due to the Central Limit Theorem, our sample mean will be normally distributed:

Therefore we can take a z-score of our sampled mean as:

This z-score will then be a simulated value from a standard normal distribution.



General Machine Learning Questions
Difference between convex and non-convex cost function; what does it mean when a cost function is non-convex?

Convex: local min = global min efficient solvers strong theoretical guarantees Examples of ML algorithms:

    Linear regression/ Ridge regression, with Tikhonov regularisation
    Sparse linear regression with L1 regularisation, such as Lasso
    Support vector machines
    Parameter estimation in Linear-Gaussian time series (Kalman filter and friends)

Non-convex

    Multi local min
    Many solvers come from convex world
    Weak theoretical guarantees if any Examples of ML algorithms:
    Neural networks
    Maximum likelihood mixtures of Gaussians

What is overfitting?

https://en.wikipedia.org/wiki/Overfitting
Describe Decision Tree, SVM, Random Forest and Boosting. Talk about their advantage and disadvantages.

https://www2.isye.gatech.edu/~tzhao80/Lectures/Lecture_6.pdf
Describe the criterion for a particular model selection. Why is dimension reduction important?

http://www.stat.cmu.edu/tr/tr759/tr759.pdf
What are the assumptions for logistic and linear regression?

    Linear regression: Linearity of residuals, Independence of residuals, Normal distribution of residuals, Equal variance of residuals. http://blog.uwgb.edu/bansalg/statistics-data-analytics/linear-regression/what-are-the-four-assumptions-of-linear-regression/
    Logistic regression: Dependent variable is binary, Observations are independent of each other, Little or no multicollinearity among the independent variables, Linearity of independent variables and log odds. https://www.statisticssolutions.com/assumptions-of-logistic-regression/

Compare Lasso and Ridge Regression.

https://blog.alexlenail.me/what-is-the-difference-between-ridge-regression-the-lasso-and-elasticnet-ec19c71c9028
What’s the difference between MLE and MAP inference?

https://wiseodd.github.io/techblog/2017/01/01/mle-vs-map/
How does K-means work? What kind of distance metric would you choose? What if different features have different dynamic range?

    Explain why and pseudo-code: http://stanford.edu/~cpiech/cs221/handouts/kmeans.html
    Distance metrics: Euclidean distance, Manhatan distance, https://pdfs.semanticscholar.org/a630/316f9c98839098747007753a9bb6d05f752e.pdf
    Explain normalization for K-means and different results you can have: https://www.edupristine.com/blog/k-means-algorithm

How many topic modeling techniques do you know of? Formulate LSI and LDA techniques.

https://towardsdatascience.com/2-latent-methods-for-dimension-reduction-and-topic-modeling-20ff6d7d547
What are generative and discriminative algorithms? What are their strengths and weaknesses? Which type of algorithms are usually used and why?”

https://cedar.buffalo.edu/~srihari/CSE574/Discriminative-Generative.pdf
Why scaling of the input is important? For which learning algorithms this is important? What is the problem with Min-Max scaling?

https://sebastianraschka.com/Articles/2014_about_feature_scaling.html
How can you plot ROC curves for multiple classes?

With macro-averaging of weights where PRE = (PRE1 + PRE2 + --- + PREk )/K https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
2.neural-networks-questions.md
Neural Networks
Is random weight assignment better than assigning same weights to the units in the hidden layer?

Because of the symmetry problem, all the units will get the same values during the forward propagation. This also will bias you to a specific local minima. https://stackoverflow.com/questions/20027598/why-should-weights-of-neural-networks-be-initialized-to-random-numbers https://medium.com/usf-msds/deep-learning-best-practices-1-weight-initialization-14e5c0295b94
Why is gradient checking important?

Gradient checking can help to find bugs in a backpropagation implementation, it is done by comparing the analytical gradient and the numerical gradient computed with calculus. https://stackoverflow.com/questions/47506521/what-exactly-is-gradient-checking http://cs231n.github.io/optimization-1/
What is the loss function in a NN?

The loss function depends on the type of problem: Regression: Mean squared error Binary classification: Binary cross entropy Multiclass: Cross entropy Ranking: Hinge loss
There is a neuron in the hidden layer that always has a large error found in backpropagation. What can be the reason?

It can be either the weight transfer from the input layer to the hidden layer for that neuron is to be blamed or the activation function for the neuron should be changed. https://medium.com/@karpathy/yes-you-should-understand-backprop-e2f06eab496b
3.svm-logr-em.questions.md
SVM and Log Regression (Log R)
Difference between SVM and Log R?

http://www.cs.toronto.edu/~kswersky/wp-content/uploads/svm_vs_lr.pdf
What does LogR give ?

Posterior probability (P(y|x))
Does SVM give any probabilistic output?

http://www.cs.cornell.edu/courses/cs678/2007sp/platt.pdf
What are the support vectors in SVM?

The vectors that define the hyperplane (margin) of SVM.
Evaluation of LogR?

You can use any evaluation metric such as Precision, Recall, AUC, F1.
How does a logistic regression model know what the coefficients are?

http://www-hsc.usc.edu/~eckel/biostat2/notes/notes14.pdf
Expectation-Maximization
How's EM done?

https://stackoverflow.com/questions/11808074/what-is-an-intuitive-explanation-of-the-expectation-maximization-technique
How are the params of EM updated?

https://stackoverflow.com/questions/11808074/what-is-an-intuitive-explanation-of-the-expectation-maximization-technique
When doing an EM for GMM, how do you find the mixture weights?

I replied that for 2 Gaussians, the prior or the mixture weight can be assumed to be a Bernouli distribution. http://www.aishack.in/tutorials/expectation-maximization-gaussian-mixture-model-mixtures/
If x ~ N(0,1), what does 2x follow?

N(0,2) https://en.wikipedia.org/wiki/Sum_of_normally_distributed_random_variables
How would you sample for a GMM?

http://www.robots.ox.ac.uk/~fwood/teaching/C19_hilary_2013_2014/gmm.pdf
How to sample from a Normal Distribution with known mean and variance?

https://stats.stackexchange.com/questions/16334/how-to-sample-from-a-normal-distribution-with-known-mean-and-variance-using-a-co
4.data-science-prob-questions.md
Data Science in Production
When you have a time series data by monthly, it has large data records, how will you find out significant difference between this month and previous months values?

Many possible answers here, mine: you sample a N large enough to reduce uncertainty over the large data, then you compare with a statistical test. https://www.sas.upenn.edu/~fdiebold/Teaching104/Ch14_slides.pdf
When users are navigating through the Amazon website, they are performing several actions. What is the best way to model if their next action would be a purchase?

A sequential machine learning algorithm where you manage to keep the state of the user and predict his/her next action. Here many options are possible HMM, RNN, Bandits.
When you recommend a set of items in a horizontal manner there is a problem we call it position bias? How do you use click data without position bias?

You sample by position making them a uniform distribution.
If you can build a perfect (100% accuracy) classification model to predict some customer behaviour, what will be the problem in application?

All the problems that can happen with overfitting.
Math and Probability
How do you weight 9 marbles three times on a balance scale to select the heaviest one?

https://mattgadient.com/2013/02/03/9-marbles-and-a-weight-balance-which-is-the-heaviest-one/
Estimate the disease probability in one city given the probability is very low nationwide. Randomly asked 1000 person in this city, with all negative response (NO disease). What is the probability of disease in this city?

https://medium.com/acing-ai/interview-guide-to-probability-distributions-a6dfb08c3766
5.programming-questions.md
Programming Questions
Given a bar plot and imagine you are pouring water from the top, how to qualify how much water can be kept in the bar chart?

https://www.geeksforgeeks.org/trapping-rain-water/
Find the cumulative sum of top 10 most profitable products of the last 6 month for customers in Seattle.

Solution: heap that keeps and updates the most profitable products.
Implement circular queue using an array.

https://www.geeksforgeeks.org/circular-queue-set-1-introduction-array-implementation/
Given a ‘csv’ file with ID and Quantity columns, 50 million records and size of data as 2 GBs, write a program in any language of your choice to aggregate the QUANTITY column.

Grep like solution, careful with overflow!
Given a function with inputs — an array with N randomly sorted numbers, and an int K, return output in an array with the K largest numbers.

https://www.geeksforgeeks.org/kth-smallestlargest-element-unsorted-array/
Given two strings, print all the inter-leavings of the Strings in which characters from two strings should be in same order as they were in original strings.

e.g. for "abc", "de", print all of these: adebc, abdec, adbce, deabc, dabce, etc, etc

https://gist.github.com/geraldyeo/6c4eaea8a1a6bcc480cac5328cbff664



ML System Design

Best way to practice is reading/watching case studies and examples on Blogs and YT.

ML Systems Design Interview Guide · Patrick Halina

Really Good Compilation

chiphuyen/machine-learning-systems-design: A booklet on machine learning systems design with exercises
Good Read

DropBox OCR Pipeline

Machine Learning System Design (YouTube Recommendation System)

Compass

Airbnb
Recommender Systems

Very high chances that a design question is related to Recommender System.

In-Depth Guide: How Recommender Systems Work | Built In

How to Design and Build a Recommendation System Pipeline in Python (Jill Cates)
Interview Questions

https://github.com/Sroy20/machine-learning-interview-questions

https://github.com/ShuaiW/data-science-question-answer

https://github.com/andrewekhalel/MLQuestions

https://github.com/iamtodor/data-science-interview-questions-and-answers

https://github.com/kojino/120-Data-Science-Interview-Questions

https://www.itshared.org/2015/10/data-science-interview-questions.html


# <a name="ml-sys"></a>  5. Machine Learning System Design

## Designing ML systems for production
This is one of my favorite interviews in which you can shine bright and up-level your career. I'd like to mention the following important notes:

- Remember, the goal of ML system design interview is NOT to measure your deep and detailed knowledge of different ML algorithms, but your ability to zoom out and design a production-level ML system that can be deployed as a service within a company's ML infrastructure.

- Deploying deep learning models in production can be challenging, and it is beyond training models with good performance. Several distinct components need to be designed and developed in order to deploy a production level deep learning system.
<p align="center">
<img src="https://github.com/alirezadir/Production-Level-Deep-Learning/blob/master/images/components.png" title="" width="70%" height="70%">
</p>

- For more insight on different components above you can check out the following resources):
  - [Full Stack Deep Learning course](https://fall2019.fullstackdeeplearning.com/)
  - [Production Level Deep Learning](https://github.com/alirezadir/Production-Level-Deep-Learning)
  - [Machine Learning Systems Design](https://github.com/chiphuyen/machine-learning-systems-design)
  - Stanford course on ML system design [TBA]

Once you learn about the basics, I highly recommend checking out different companies blogs on ML systems, which I learnt a lot from. You can refer to some of those resources in the subsection [ML at Companies](#ml-at-companies) below.

## ML System Design Flow
Approaching an ML system design problem follows a similar flow to the generic software system design.
For more insight on general system design interview you can e.g. check out:
- [Grokking the System Design Interview
](https://www.educative.io/courses/grokking-the-system-design-interview)
- [System design primer](https://github.com/donnemartin/system-design-primer)
- [Deep Learning Interviews](https://www.amazon.in/Deep-Learning-Interviews-interview-questions/dp/1916243568)

Below is a design flow that I would recommend:

1. Problem Description
    - What does it mean? 
    - Use cases 
    - Requirements
    - Assumptions 
2. Do we need ML to solve this problem? 
    - Trade off between impact and cost
      - Costs: Data collection, data annotation, compute 
    - if Yes, go to the next topic. If No, follow a general system design flow. 
3. ML Metrics 
      - Accuracy metrics: 
          - imbalanced data?
      - Latency 
      - Problem specific metric (e.g. CTR)
4. Data
    - Needs 
        - type (e.g. image, text, video, etc) and volume
    - Sources
        - availability and cost 
    - Labelling (if needed)
      - labeling cost  
5. MVP Logic 
    - Model based vs rule based logic 
        - Pros and cons, and decision 
          -  Note: Always start as simple as possible and iterate over 
    - Propose a simple model (e.g. a binary logistic regression classifier)
    - Features/ Signals (if needed)
      - what to chose as and how to chose features 
      - feature representation 
6. Training (if needed)
      - data splits (train, dev, test)
        - portions
        - how to chose a test set 
      - debugging 
    - Iterate over MVP model (if needed)
      - data augmentation  
7. Inference (online)
    - Data processing and verification 
    - Prediction module 
    - Serving infra 
    - Web app 
8. Scaling
  - Scaling for increased demand (same as in distributed systems)
      - Scaling web app and serving system 
      - Data partitioning 
  - Data parallelism 
  - Model parallelism 
9. A/B test and deployment
    - How to A/B test? 
      - what portion of users?
      - control and test groups 
10. Monitoring and Updates 
    - seasonality   


## ML System Design Topics
I observed there are certain sets of topics that are frequently brought up or can be used as part of the logic of the system. Here are some of the important ones:

### Recommendation Systems
- Collaborative Filtering (CF)
    - user based, item based
    - Cold start problem
    - Matrix factorization
- Content based filtering

### NLP

- Preprocessing
  - Normalization, tokenization, stop words
- Word Embeddings
  - Word2Vec, GloVe, Elmo, BERT
- Text classification and sentiment analysis
- NLP specialist topics:
  - Language Modeling
  - Part of speech tagging
  - POS HMM
    - Viterbi algorithm and beam search
  - Named entity recognition
  - Topic modeling
  - Speech Recognition Systems
    - Feature extraction, MFCCs
    - Acoustic modeling
      - HMMs for AM
      - CTC algorithm (advanced)
    - Language modeling
      - N-grams vs deep learning models (trade-offs)
      - Out of vocabulary problem
  - Dialog and chatbots
    - [CMU lecture on chatbots](http://tts.speech.cs.cmu.edu/courses/11492/slides/chatbots_shrimai.pdf)
    - [CMU lecture on spoken dialogue systems](http://tts.speech.cs.cmu.edu/courses/11492/slides/sds_components.pdf)
  - Machine Translation
    - Seq2seq models, NMT

Note: The reason I have more topics here is because this was my focus in my own interviews

### Ads and Ranking
- CTR prediction
- Ranking algorithms

### Information retrieval
- Search
  - Pagerank
  - Autocomplete for search

### Computer vision
- Image classification
- Object Tracking
- Popular architectures (AlexNet, VGG, ResNET)
- [TBD]

### Transfer learning
- Why and when to use transfer learning
- How to do it
  - depending on the dataset sizes and similarities


## ML Systems at Big Companies 
- AI at LinkedIn
  - [Intro to AI at Linkedin](https://engineering.linkedin.com/blog/2018/10/an-introduction-to-ai-at-linkedin)
  - [Building The LinkedIn Knowledge Graph](https://engineering.linkedin.com/blog/2016/10/building-the-linkedin-knowledge-graph)
  - [The AI Behind LinkedIn Recruiter search and recommendation systems](https://engineering.linkedin.com/blog/2019/04/ai-behind-linkedin-recruiter-search-and-recommendation-systems)
  - [A closer look at the AI behind course recommendations on LinkedIn Learning, Part 1](https://engineering.linkedin.com/blog/2020/course-recommendations-ai-part-one)
  - [A closer look at the AI behind course recommendations on LinkedIn Learning, Part 2](https://engineering.linkedin.com/blog/2020/course-recommendations-ai-part-two)
  - [Communities AI: Building communities around interests on LinkedIn](https://engineering.linkedin.com/blog/2019/06/building-communities-around-interests)
  - [Linkedin's follow feed](https://engineering.linkedin.com/blog/2016/03/followfeed--linkedin-s-feed-made-faster-and-smarter)
  - XNLT for A/B testing


- ML at Google
    - ML pipelines with TFX and KubeFlow
    - [How Google Search works](https://www.google.com/search/howsearchworks/)
      - Page Rank algorithm ([intro to page rank](https://www.youtube.com/watch?v=IKXvSKaI2Ko), [the algorithm that started google](https://www.youtube.com/watch?v=qxEkY8OScYY))
    - TFX production components
      - [TFX workshop by Robert Crowe](https://conferences.oreilly.com/artificial-intelligence/ai-ca-2019/cdn.oreillystatic.com/en/assets/1/event/298/TFX_%20Production%20ML%20pipelines%20with%20TensorFlow%20Presentation.pdf)
    - [Google Cloud Platform Big Data and Machine Learning Fundamentals](https://www.coursera.org/learn/gcp-big-data-ml-fundamentals)
- Scalable ML using AWS
  - [AWS Machine Learning Blog](https://aws.amazon.com/blogs/machine-learning/)
  - [Deploy a machine learning model with AWS Elastic Beanstalk](https://medium.com/swlh/deploy-a-machine-learning-model-with-aws-elasticbeanstalk-dfcc47b6043e)
  - [Deploying Machine Learning Models as API using AWS](https://medium.com/towards-artificial-intelligence/deploying-machine-learning-models-as-api-using-aws-a25d05518084)
  - [Serverless Machine Learning On AWS Lambda](https://medium.com/swlh/how-to-deploy-your-scikit-learn-model-to-aws-44aabb0efcb4)
-  ML at Facebook
   -  [Machine Learning at Facebook Talk](https://www.youtube.com/watch?v=C4N1IZ1oZGw)
   -  [Scaling AI Experiences at Facebook with PyTorch](https://www.youtube.com/watch?v=O8t9xbAajbY)
   -  [Understanding text in images and videos](https://ai.facebook.com/blog/rosetta-understanding-text-in-images-and-videos-with-machine-learning/)
   -  [Protecting people](https://ai.facebook.com/blog/advances-in-content-understanding-self-supervision-to-protect-people/)
   -  Ads
      - Ad CTR prediction
      - [Practical Lessons from Predicting Clicks on Ads at Facebook](https://quinonero.net/Publications/predicting-clicks-facebook.pdf)
   - Newsfeed Ranking
     - [How Facebook News Feed Works](https://techcrunch.com/2016/09/06/ultimate-guide-to-the-news-feed/)
     - [How does Facebook’s advertising targeting algorithm work?](https://quantmar.com/99/How-does-facebooks-advertising-targeting-algorithm-work)
     - [ML and Auction Theory](https://www.youtube.com/watch?v=94s0yYECeR8)
     - [Serving Billions of Personalized News Feeds with AI - Meihong Wang](https://www.youtube.com/watch?v=wcVJZwO_py0&t=80s)
     - [Generating a Billion Personal News Feeds](https://www.youtube.com/watch?v=iXKR3HE-m8c&list=PLefpqz4O1tblTNAtKaSIOU8ecE6BATzdG&index=2)
     - [Instagram feed ranking](https://www.facebook.com/atscaleevents/videos/1856120757994353/?v=1856120757994353)
     - [How Instagram Feed Works](https://techcrunch.com/2018/06/01/how-instagram-feed-works/)
   - [Photo search](https://engineering.fb.com/ml-applications/under-the-hood-photo-search/)
   - Social graph search
   - Recommendation
     - [Recommending items to more than a billion people](https://engineering.fb.com/core-data/recommending-items-to-more-than-a-billion-people/)
     - [Social recommendations](https://engineering.fb.com/android/made-in-ny-the-engineering-behind-social-recommendations/)
   - [Live videos](https://engineering.fb.com/ios/under-the-hood-broadcasting-live-video-to-millions/)
   - [Large Scale Graph Partitioning](https://engineering.fb.com/core-data/large-scale-graph-partitioning-with-apache-giraph/)
   - [TAO: Facebook’s Distributed Data Store for the Social Graph](https://www.youtube.com/watch?time_continue=66&v=sNIvHttFjdI&feature=emb_logo) ([Paper](https://www.usenix.org/system/files/conference/atc13/atc13-bronson.pdf))
   - [NLP at Facebook](https://www.youtube.com/watch?v=ZcMvffdkSTE)
-  ML at Netflix
   -  [Recommendation at Netflix](https://www.slideshare.net/moustaki/recommending-for-the-world)
   -  [Past, Present & Future of Recommender Systems: An Industry Perspective](https://www.slideshare.net/justinbasilico/past-present-future-of-recommender-systems-an-industry-perspective)
   -  [Deep learning for recommender systems](https://www.slideshare.net/moustaki/deep-learning-for-recommender-systems-86752234)
   -  [Reliable ML at Netflix](https://www.slideshare.net/justinbasilico/making-netflix-machine-learning-algorithms-reliable)
   -  [ML at Netflix (Spark and GraphX)](https://www.slideshare.net/SessionsEvents/ehtsham-elahi-senior-research-engineer-personalization-science-and-engineering-group-at-netflix-at-mlconf-sea-50115?next_slideshow=1)
   -  [Recent Trends in Personalization](https://www.slideshare.net/justinbasilico/recent-trends-in-personalization-a-netflix-perspective)
   -  [Artwork Personalization @ Netflix](https://www.slideshare.net/justinbasilico/artwork-personalization-at-netflix)

# <a name="breadth"></a> 4. ML Breadth/Fundamentals
As the name suggests, this interview is intended to evaluate your general knowledge of ML concepts both from theoretical and practical perspectives. Unlike ML depth interviews, the breadth interviews tend to follow a pretty similar structure and coverage amongst different interviewers and interviewees.

The best way to prepare for this interview is to review your notes from ML courses as well some high quality online courses and material. In particular, I found the following resources pretty helpful.

## Courses and review material:
- [Andrew Ng's Machine Learning Course](https://www.coursera.org/learn/machine-learning) (you can also find the [lectures on Youtube](https://www.youtube.com/watch?v=PPLop4L2eGk&list=PLLssT5z_DsK-h9vYZkQkYNWcItqhlRJLN) )
- [Structuring Machine Learning Projects](https://www.coursera.org/learn/machine-learning-projects)
- [Udacity's deep learning nanodegree](https://www.udacity.com/course/deep-learning-nanodegree--nd101) or  [Coursera's Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning) (for deep learning)


If you already know the concepts, the following resources are pretty useful for a quick review of different concepts:
- [StatQuest Machine Learning videos](https://www.youtube.com/watch?v=Gv9_4yMHFhI&list=PLblh5JKOoLUICTaGLRoHQDuF_7q2GfuJF)
- [StatQuest Statistics](https://www.youtube.com/watch?v=qBigTkBLU6g&list=PLblh5JKOoLUK0FLuzwntyYI10UQFUhsY9) (for statistics review - most useful for Data Science roles)
- [Machine Learning cheatsheets](https://ml-cheatsheet.readthedocs.io/en/latest/)
- [Chris Albon's ML falshcards](https://machinelearningflashcards.com/)

Below are the most important topics to cover:
## 1. Classic ML Concepts

### ML Algorithms' Categories
  - Supervised, unsupervised, and semi-supervised learning (with examples)
    - Classification vs regression vs clustering
  - Parametric vs non-parametric algorithms
  - Linear vs Nonlinear algorithms
### Supervised learning
  - Linear Algorithms
    - Linear regression
      - least squares, residuals,  linear vs multivariate regression
    - Logistic regression
      - cost function (equation, code),  sigmoid function, cross entropy
    - Support Vector Machines
    - Linear discriminant analysis

  - Decision Trees
    - Logits
    - Leaves
    - Training algorithm
      - stop criteria
    - Inference
    - Pruning

  - Ensemble methods
    - Bagging and boosting methods (with examples)
    - Random Forest
    - Boosting
      - Adaboost
      - GBM
      - XGBoost
  - Comparison of different algorithms
    - [TBD: LinkedIn lecture]

  - Optimization
    - Gradient descent (concept, formula, code)
    - Other variations of gradient descent
      - SGD
      - Momentum
      - RMSprop
      - ADAM
  - Loss functions
    - Logistic Loss function 
    - Cross Entropy (remember formula as well)
    - Hinge loss (SVM)

- Feature selection
  - Feature importance
- Model evaluation and selection
  - Evaluation metrics
    - TP, FP, TN, FN
    - Confusion matrix
    - Accuracy, precision, recall/sensitivity, specificity, F-score
      - how do you choose among these? (imbalanced datasets)
      - precision vs TPR (why precision)
    - ROC curve (TPR vs FPR, threshold selection)
    - AUC (model comparison)
    - Extension of the above to multi-class (n-ary) classification
    - algorithm specific metrics [TBD]
  - Model selection
    - Cross validation
      - k-fold cross validation (what's a good k value?)

### Unsupervised learning
  - Clustering
    - Centroid models: k-means clustering
    - Connectivity models: Hierarchical clustering
    - Density models: DBSCAN
  - Gaussian Mixture Models
  - Latent semantic analysis
  - Hidden Markov Models (HMMs)
    - Markov processes
    - Transition probability and emission probability
    - Viterbi algorithm [Advanced]
  - Dimension reduction techniques
    - Principal Component Analysis (PCA)
    - Independent Component Analysis (ICA)
    - T-sne


### Bias / Variance (Underfitting/Overfitting)
- Regularization techniques
  - L1/L2 (Lasso/Ridge)
### Sampling
- sampling techniques
  - Uniform sampling
  - Reservoir sampling
  - Stratified sampling
### Missing data
 - [TBD]
### Time complexity of ML algorithms
- [TBD]

## 2. Deep learning
- Feedforward NNs
  - In depth knowledge of how they work
  - [EX] activation function for classes that are not mutually exclusive
- RNN
  - backpropagation through time (BPTT)
  - vanishing/exploding gradient problem
- LSTM
  - vanishing/exploding gradient problem
  -  gradient?
- Dropout
  - how to apply dropout to LSTM?
- Seq2seq models
- Attention
  - self-attention
- Transformer and its architecture (in details, yes, no kidding! I was asked twice! In an ideal world, I wouldn't answer those detailed questions to anyone except the authors and teammates, as either you've designed it or memorized it!)
- Embeddings (word embeddings)


## 3. Statistical ML
###  Bayesian algorithms
  - Naive Bayes
  - Maximum a posteriori (MAP) estimation
  - Maximum Likelihood (ML) estimation
### Statistical significance
- R-squared
- P-values

## 4. Other topics:
  - Outliers
  - Similarity/dissimilarity metrics
    - Euclidean, Manhattan, Cosine, Mahalanobis (advanced)


# Deep Learning - List of questions

## Table of Contents

- [General questions](#general-questions)
- [Machine Learning basics](#machine-learning-basics)
- [Optimization procedures](#optimization-procedures)
- [Parameter initialization](#parameter-initialization)
- [Sequence Modeling](#sequence-modeling)
- [Autoencoders](#autoencoders)
- [Representation Learning](#representation-learning)
- [Monte Carlo Methods](#monte-carlo-methods)
- [Adversarial Networks](#adversarial-networks)

## General questions

1. How will you implement dropout during forward and backward pass?
1. What do you do if Neural network training loss/testing loss stays constant? (ask if there could be an error in your code, going deeper, going simpler…)
1. Why do RNNs have a tendency to suffer from exploding/vanishing gradient? How to prevent this? (Talk about LSTM cell which helps the gradient from vanishing, but make sure you know why it does so. Talk about gradient clipping, and discuss whether to clip the gradient element wise, or clip the norm of the gradient.)
1. Do you know GAN, VAE, and memory augmented neural network? Can you talk about it?
1. Does using full batch means that the convergence is always better given unlimited power? (Beautiful explanation by Alex Seewald: https://www.quora.com/Is-full-batch-gradient-descent-with-unlimited-computer-power-always-better-than-mini-batch-gradient-descent)
1. What is the problem with sigmoid during backpropagation? (Very small, between 0.25 and zero.)
1. Given a black box machine learning algorithm that you can’t modify, how could you improve its error? (you can transform the input for example.)
1. How to find the best hyper parameters? (Random search, grid search, Bayesian search (and what it is?))
1. What is transfer learning?
1. Compare and contrast L1-loss vs. L2-loss and L1-regularization vs. L2-regularization.

## Machine Learning basics

1. Can you state Tom Mitchell's definition of learning and discuss T, P and E?
1. What can be different types of tasks encountered in Machine Learning?
1. What are supervised, unsupervised, semi-supervised, self-supervised, multi-instance learning, and reinforcement learning?
1. Loosely how can supervised learning be converted into unsupervised learning and vice-versa?
1. Consider linear regression. What are T, P and E?
1. Derive the normal equation for linear regression.
1. What do you mean by affine transformation? Discuss affine vs. linear transformation.
1. Discuss training error, test error, generalization error, overfitting, and underfitting.
1. Compare representational capacity vs. effective capacity of a model.
1. Discuss VC dimension.
1. What are nonparametric models? What is nonparametric learning?
1. What is an ideal model? What is Bayes error? What is/are the source(s) of Bayes error occur?
1. What is the no free lunch theorem in connection to Machine Learning?
1. What is regularization? Intuitively, what does regularization do during the optimization procedure? (expresses preferences to certain solutions, implicitly and explicitly)
1. What is weight decay? What is it added?
1. What is a hyperparameter? How do you choose which settings are going to be hyperparameters and which are going to be learnt? (either difficult to optimize or not appropriate to learn - learning model capacity by learning the degree of a polynomial or coefficient of the weight decay term always results in choosing the largest capacity until it overfits on the training set)
1. Why is a validation set necessary?
1. What are the different types of cross-validation? When do you use which one?
1. What are point estimation and function estimation in the context of Machine Learning? What is the relation between them?
1. What is the maximal likelihood of a parameter vector $theta$? Where does the log come from?
1. Prove that for linear regression MSE can be derived from maximal likelihood by proper assumptions.
1. Why is maximal likelihood the preferred estimator in ML? (consistency and efficiency)
1. Under what conditions do the maximal likelihood estimator guarantee consistency?
1. What is cross-entropy of loss? (trick question) 


## Optimization procedures

1. What is the difference between an optimization problem and a Machine Learning problem?
1. How can a learning problem be converted into an optimization problem?
1. What is empirical risk minimization? Why the term empirical? Why do we rarely use it in the context of deep learning?
1. Name some typical loss functions used for regression. Compare and contrast. (L2-loss, L1-loss, and Huber loss)
1. What is the 0-1 loss function? Why can't the 0-1 loss function or classification error be used as a loss function for optimizing a deep neural network? (Non-convex, gradient is either 0 or undefined. https://davidrosenberg.github.io/ml2015/docs/3a.loss-functions.pdf)
1. 


## Parameter initialization

1.  

## Sequence Modeling

1. Write the equation describing a dynamical system. Can you unfold it? Now, can you use this to describe a RNN? (include hidden, input, output, etc.) 
1. What determines the size of an unfolded graph?
1. What are the advantages of an unfolded graph? (arbitrary sequence length, parameter sharing, and illustrate information flow during forward and backward pass)
1. What does the output of the hidden layer of a RNN at any arbitrary time _t_ represent?
1. Are the output of hidden layers of RNNs lossless? If not, why?
1. RNNs are used for various tasks. From a RNNs point of view, what tasks are more demanding than others? 
1. Discuss some examples of important design patterns of classical RNNs.
1. Write the equations for a classical RNN where hidden layer has recurrence. How would you define the loss in this case? What problems you might face while training it? (Discuss runtime)
1. What is backpropagation through time? (BPTT)
1. Consider a RNN that has only output to hidden layer recurrence. What are its advantages or disadvantages compared to a RNNhaving only hidden to hidden recurrence? 
1. What is Teacher forcing? Compare and contrast with BPTT.
1. What is the disadvantage of using a strict teacher forcing technique? How to solve this?
1. 
1. Explain the vanishing/exploding gradient phenomenon for recurrent neural networks. (use scalar and vector input scenarios)
1. Why don't we see the vanishing/exploding gradient phenomenon in feedforward networks? (weights are different in different layers - Random block intialization paper)
1. What is the key difference in architecture of LSTMs/GRUs compared to traditional RNNs? (Additive update instead of multiplicative)
1. What is the difference between LSTM and GRU?
1. Explain Gradient Clipping. 
1. Adam and RMSProp adjust the size of gradients based on previously seen gradients. Do they inherently perform gradient clipping? If no, why?
1. Discuss RNNs in the context of Bayesian Machine Learning.
1. Can we do Batch Normalization in RNNs? If not, what is the alternative? (BNorm would need future data; Layer Norm)

## Autoencoders

1. What is an Autoencoder? What does it "auto-encode"?
1. What were Autoencoders traditionally used for? Why there has been a resurgence of Autoencoders for generative modeling?
1. What is recirculation?
1. What loss functions are used for Autoencoders?
1. What is a linear autoencoder? Can it be optimal (lowest training reconstruction error)? If yes, under what conditions?
1. What is the difference between Autoencoders and PCA (can also be used for reconstruction - https://stats.stackexchange.com/questions/229092/how-to-reverse-pca-and-reconstruct-original-variables-from-several-principal-com).
1. What is the impact of the size of the hidden layer in Autoencoders?
1. What is an undercomplete Autoencoder? Why is it typically used for? 
1. What is a linear Autoencoder? Discuss it's equivalence with PCA. (only valid for undercomplete) Which one is better in reconstruction?
1. What problems might a nonlinear undercomplete Autoencoder face?
1. What are overcomplete Autoencoders? What problems might they face? Does the scenario change for linear overcomplete autoencoders? (identity function)
1. Discuss the importance of regularization in the context of Autoencoders.
1. Why does generative autoencoders not require regularization?
1. What are sparse autoencoders?
1. What is a denoising autoencoder? What are its advantages? How does it solve the overcomplete problem?
1. What is score matching? Discuss it's connections to DAEs.
1. Are there any connections between Autoencoders and RBMs?
1. What is manifold learning? How are denoising and contractive autoencoders equipped to do manifold learning? 
1. What is a contractive autoencoder? Discuss its advantages. How does it solve the overcomplete problem?
1. Why is a contractive autoencoder named so? (intuitive and mathematical)
1. What are the practical issues with CAEs? How to tackle them?
1. What is a stacked autoencoder? What is a deep autoencoder? Compare and contrast.
1. Compare the reconstruction quality of a deep autoencoder vs. PCA.
1. What is predictive sparse decomposition?
1. Discuss some applications of Autoencoders.

## Representation Learning

1. What is representation learning? Why is it useful? (for a particular architecture, for other tasks, etc.) 
1. What is the relation between Representation Learning and Deep Learning?
1. What is one-shot and zero-shot learning (Google's NMT)? Give examples.
1. What trade offs does representation learning have to consider?
1. What is greedy layer-wise unsupervised pretraining (GLUP)? Why greedy? Why layer-wise? Why unsupervised? Why pretraining?
1. What were/are the purposes of the above technique? (deep learning problem and initialization)
1. Why does unsupervised pretraining work?
1. When does unsupervised training work? Under which circumstances?
1. Why might unsupervised pretraining act as a regularizer?
1. What is the disadvantage of unsupervised pretraining compared to other forms of unsupervised learning?
1. How do you control the regularizing effect of unsupervised pretraining?
1. How to select the hyperparameters of each stage of GLUP?
1. 

## Monte Carlo Methods

1. What are deterministic algorithms? (nothing random)
2. What are Las vegas algorithms? (exact or no solution, random resources)
3. What are deterministic approximate algorithms? (solution is not exact but the error is known)
4. What are Monte Carlo algorithms? (approximate solution with random error)
5. 


## Adversarial Networks

1. Discuss state-of-the-art attack and defense techniques for adversarial models.
1.  

# Machine Learning - List of questions

## Learning Theory

1. Describe bias and variance with examples. 
1. What is Empirical Risk Minimization? 
1. What is Union bound and Hoeffding's inequality? 
1. Write the formulae for training error and generalization error. Point out the differences.
1. State the uniform convergence theorem and derive it. 
1. What is sample complexity bound of uniform convergence theorem? 
1. What is error bound of uniform convergence theorem? 
1. What is the bias-variance trade-off theorem? 
1. From the bias-variance trade-off, can you derive the bound on training set size?
1. What is the VC dimension? 
1. What does the training set size depend on for a finite and infinite hypothesis set? Compare and contrast. 
1. What is the VC dimension for an n-dimensional linear classifier? 
1. How is the VC dimension of a SVM bounded although it is projected to an infinite dimension? 
1. Considering that Empirical Risk Minimization is a NP-hard problem, how does logistic regression and SVM loss work? 

## Model and feature selection
1. Why are model selection methods needed?
1. How do you do a trade-off between bias and variance?
1. What are the different attributes that can be selected by model selection methods?
1. Why is cross-validation required?
1. Describe different cross-validation techniques.
1. What is hold-out cross validation? What are its advantages and disadvantages?
1. What is k-fold cross validation? What are its advantages and disadvantages?
1. What is leave-one-out cross validation? What are its advantages and disadvantages?
1. Why is feature selection required?
1. Describe some feature selection methods.
1. What is forward feature selection method? What are its advantages and disadvantages?
1. What is backward feature selection method? What are its advantages and disadvantages?
1. What is filter feature selection method and describe two of them?
1. What is mutual information and KL divergence?
1. Describe KL divergence intuitively.

## Curse of dimensionality 
1. Describe the curse of dimensionality with examples.
1. What is local constancy or smoothness prior or regularization?

## Universal approximation of neural networks
1. State the universal approximation theorem? What is the technique used to prove that?
1. What is a Borel measurable function?
1. Given the universal approximation theorem, why can't a MLP still reach a arbitrarily small positive error?

## Deep Learning motivation
1. What is the mathematical motivation of Deep Learning as opposed to standard Machine Learning techniques?
1. In standard Machine Learning vs. Deep Learning, how is the order of number of samples related to the order of regions that can be recognized in the function space?
1. What are the reasons for choosing a deep model as opposed to shallow model? (1. Number of regions O(2^k) vs O(k) where k is the number of training examples 2. # linear regions carved out in the function space depends exponentially on the depth. )
1. How Deep Learning tackles the curse of dimensionality? 

## Support Vector Machine
1. How can the SVM optimization function be derived from the logistic regression optimization function?
1. What is a large margin classifier?
1. Why SVM is an example of a large margin classifier?
1. SVM being a large margin classifier, is it influenced by outliers? (Yes, if C is large, otherwise not)
1. What is the role of C in SVM?
1. In SVM, what is the angle between the decision boundary and theta?
1. What is the mathematical intuition of a large margin classifier?
1. What is a kernel in SVM? Why do we use kernels in SVM?
1. What is a similarity function in SVM? Why it is named so?
1. How are the landmarks initially chosen in an SVM? How many and where?
1. Can we apply the kernel trick to logistic regression? Why is it not used in practice then?
1. What is the difference between logistic regression and SVM without a kernel? (Only in implementation – one is much more efficient and has good optimization packages)
1. How does the SVM parameter C affect the bias/variance trade off? (Remember C = 1/lambda; lambda increases means variance decreases)
1. How does the SVM kernel parameter sigma^2 affect the bias/variance trade off?
1. Can any similarity function be used for SVM? (No, have to satisfy Mercer’s theorem)
1. Logistic regression vs. SVMs: When to use which one? 
( Let's say n and m are the number of features and training samples respectively. If n is large relative to m use log. Reg. or SVM with linear kernel, If n is small and m is intermediate, SVM with Gaussian kernel, If n is small and m is massive, Create or add more fetaures then use log. Reg. or SVM without a kernel)

## Bayesian Machine Learning
1. What are the differences between “Bayesian” and “Freqentist” approach for Machine Learning?
1. Compare and contrast maximum likelihood and maximum a posteriori estimation.
1. How does Bayesian methods do automatic feature selection?
1. What do you mean by Bayesian regularization?
1. When will you use Bayesian methods instead of Frequentist methods? (Small dataset, large feature set)

## Regularization
1. What is L1 regularization?
1. What is L2 regularization?
1. Compare L1 and L2 regularization.
1. Why does L1 regularization result in sparse models? [here](https://stats.stackexchange.com/questions/45643/why-l1-norm-for-sparse-models)

## Evaluation of Machine Learning systems
1. What are accuracy, sensitivity, specificity, ROC?
1. What are precision and recall?
1. Describe t-test in the context of Machine Learning.

## Clustering
1. Describe the k-means algorithm.
1. What is distortion function? Is it convex or non-convex?
1. Tell me about the convergence of the distortion function.
1. Topic: EM algorithm
1. What is the Gaussian Mixture Model?
1. Describe the EM algorithm intuitively. 
1. What are the two steps of the EM algorithm
1. Compare GMM vs GDA.

## Dimensionality Reduction
1. Why do we need dimensionality reduction techniques? (data compression, speeds up learning algorithm and visualizing data)
1. What do we need PCA and what does it do? (PCA tries to find a lower dimensional surface such the sum of the squared projection error is minimized)
1. What is the difference between logistic regression and PCA?
1. What are the two pre-processing steps that should be applied before doing PCA? (mean normalization and feature scaling)

## Basics of Natural Language Processing
1. What is WORD2VEC?
1. What is t-SNE? Why do we use PCA instead of t-SNE?
1. What is sampled softmax?
1. Why is it difficult to train a RNN with SGD?
1. How do you tackle the problem of exploding gradients? (By gradient clipping)
1. What is the problem of vanishing gradients? (RNN doesn't tend to remember much things from the past)
1. How do you tackle the problem of vanishing gradients? (By using LSTM)
1. Explain the memory cell of a LSTM. (LSTM allows forgetting of data and using long memory when appropriate.)
1. What type of regularization do one use in LSTM?
1. What is Beam Search?
1. How to automatically caption an image? (CNN + LSTM)

## Miscellaneous
1. What is the difference between loss function, cost function and objective function?


# Mathematics - List of questions

## Linear Algebra
1. What is broadcasting in connection to Linear Algebra?
1. What are scalars, vectors, matrices, and tensors?
1. What is Hadamard product of two matrices?
1. What is an inverse matrix?
1. If inverse of a matrix exists, how to calculate it?
1. What is the determinant of a square matrix? How is it calculated (Laplace expansion)? What is the connection of determinant to eigenvalues?
1. Discuss span and linear dependence.
1. What is Ax = b? When does Ax =b has a unique solution? 
1. In Ax = b, what happens when A is fat or tall?
1. When does inverse of A exist?
1. What is a norm? What is L1, L2 and L infinity norm?
1. What are the conditions a norm has to satisfy?
1. Why is squared of L2 norm preferred in ML than just L2 norm?
1. When L1 norm is preferred over L2 norm?
1. Can the number of nonzero elements in a vector be defined as L0 norm? If no, why?
1. What is Frobenius norm?
1. What is a diagonal matrix? (D_i,j = 0 for i != 0)
1. Why is multiplication by diagonal matrix computationally cheap? How is the multiplication different for square vs. non-square diagonal matrix?
1. At what conditions does the inverse of a diagonal matrix exist? (square and all diagonal elements non-zero)
1. What is a symmetrix matrix? (same as its transpose)
1. What is a unit vector?
1. When are two vectors x and y orthogonal? (x.T * y = 0)
1. At R^n what is the maximum possible number of orthogonal vectors with non-zero norm?
1. When are two vectors x and y orthonormal? (x.T * y = 0 and both have unit norm)
1. What is an orthogonal matrix? Why is computationally preferred? (a square matrix whose rows are mutually orthonormal and columns are mutually orthonormal.)
1. What is eigendecomposition, eigenvectors and eigenvalues?
1. How to find eigen values of a matrix?
1. Write the eigendecomposition formula for a matrix. If the matrix is real symmetric, how will this change?
1. Is the eigendecomposition guaranteed to be unique? If not, then how do we represent it?
1. What are positive definite, negative definite, positive semi definite and negative semi definite matrices?
1. What is SVD? Why do we use it? Why not just use ED?
1. Given a matrix A, how will you calculate its SVD?
1. What are singular values, left singulars and right singulars?
1. What is the connection of SVD of A with functions of A?
1. Why are singular values always non-negative?
1. What is the Moore Penrose pseudo inverse and how to calculate it?
1. If we do Moore Penrose pseudo inverse on Ax = b, what solution is provided is A is fat? Moreover, what solution is provided if A is tall?
1. Which matrices can be decomposed by ED? (Any NxN square matrix with N linearly independent eigenvectors)
1. Which matrices can be decomposed by SVD? (Any matrix; V is either conjugate transpose or normal transpose depending on whether A is complex or real)
1. What is the trace of a matrix?
1. How to write Frobenius norm of a matrix A in terms of trace?
1. Why is trace of a multiplication of matrices invariant to cyclic permutations?
1. What is the trace of a scalar?
1. Write the frobenius norm of a matrix in terms of trace?

## Numerical Optimization
1. What is underflow and overflow? 
1. How to tackle the problem of underflow or overflow for softmax function or log softmax function? 
1. What is poor conditioning? 
1. What is the condition number? 
1. What are grad, div and curl?
1. What are critical or stationary points in multi-dimensions?
1. Why should you do gradient descent when you want to minimize a function?
1. What is line search?
1. What is hill climbing?
1. What is a Jacobian matrix?
1. What is curvature?
1. What is a Hessian matrix?

## Basics of Probability and Informaion Theory
1. Compare "Frequentist probability" vs. "Bayesian probability"?
1. What is a random variable?
1. What is a probability distribution?
1. What is a probability mass function?
1. What is a probability density function?
1. What is a joint probability distribution?
1. What are the conditions for a function to be a probability mass function?
1. What are the conditions for a function to be a probability density function?
1. What is a marginal probability? Given the joint probability function, how will you calculate it?
1. What is conditional probability? Given the joint probability function, how will you calculate it?
1. State the Chain rule of conditional probabilities.
1. What are the conditions for independence and conditional independence of two random variables?
1. What are expectation, variance and covariance?
1. Compare covariance and independence.
1. What is the covariance for a vector of random variables?
1. What is a Bernoulli distribution? Calculate the expectation and variance of a random variable that follows Bernoulli distribution?
1. What is a multinoulli distribution?
1. What is a normal distribution?
1. Why is the normal distribution a default choice for a prior over a set of real numbers?
1. What is the central limit theorem?
1. What are exponential and Laplace distribution?
1. What are Dirac distribution and Empirical distribution?
1. What is mixture of distributions?
1. Name two common examples of mixture of distributions? (Empirical and Gaussian Mixture)
1. Is Gaussian mixture model a universal approximator of densities?
1. Write the formulae for logistic and softplus function.
1. Write the formulae for Bayes rule.
1. What do you mean by measure zero and almost everywhere?
1. If two random variables are related in a deterministic way, how are the PDFs related?
1. Define self-information. What are its units?
1. What are Shannon entropy and differential entropy?
1. What is Kullback-Leibler (KL) divergence?
1. Can KL divergence be used as a distance measure?
1. Define cross-entropy.
1. What are structured probabilistic models or graphical models?
1. In the context of structured probabilistic models, what are directed and undirected models? How are they represented?
What are cliques in undirected structured probabilistic models?

## Confidence interval
1. What is population mean and sample mean?
1. What is population standard deviation and sample standard deviation?
1. Why population s.d. has N degrees of freedom while sample s.d. has N-1 degrees of freedom? In other words, why 1/N inside root for pop. s.d. and 1/(N-1) inside root for sample s.d.? (Here)
1. What is the formula for calculating the s.d. of the sample mean?
1. What is confidence interval?
1. What is standard error?

ML Interview questions

    21 Machine Learning Interview Questions and Answers https://elitedatascience.com/machine-learning-interview-questions-answers
    Glassdoor - Amazon Machine Learning Interview Questions https://www.glassdoor.com/Interview/Amazon-Machine-Learning-Interview-Questions-EI_IE6036.0,6_KO7,23.htm
    Glassdoor - LinkedIn Software Engineer/Machine Learning Interview Questions Experience https://www.glassdoor.com/Interview/LinkedIn-Interview-Questions-E34865.htm?filter.jobTitleExact=Software+Engineer%2FMachine+Learning
    Data Science and Machine Learning Interview Questions https://towardsdatascience.com/data-science-and-machine-learning-interview-questions-3f6207cf040b
    12 Important Machine Learning Interview Questions to Study Ahead of Time. https://www.simplilearn.com/machine-learning-interview-questions-and-answers-article
    Popular Machine Learning Interview Questions To Assess Candidates. https://analyticsindiamag.com/popular-machine-learning-interview-questions-used-to-assess-candidates/
    
    
    
    
