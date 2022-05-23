https://github.com/rbackupX/QuestionBank


https://www.mlstack.cafe/blog/tensorflow-interview-questions

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



# ML Q & A interview prep:

## ML Algoritms + Basic ML

Q: What are the various types of ML? 

A: Supervised (labeled data) , Unsupervised (unlabeled data) , Reinforcement (using penalty and reward)

Q: What is your favorite algorithm? Is it used for classification or regression? (explain in under a minute)

A: Open-ended

Q: What is the difference between ML, DL and AI?

A: - AI involves machines that can perform tasks that are characteristic of human intelligence
ML is a way of achieving AI by “training” an algorithm so that it can learn data.
Deep learning is one of many approaches to machine learning, which uses Neural Networks that mimic the biological structure of the brain. 
Another difference is the feature extraction and classification are separate steps for ML but are a single NN for DL.

Q: Talk about a recent ML paper that you’ve read in 2 minutes.

A: Open-ended

Be able to explain the following metrics: Accuracy, Precision, Recall, and F1, TP, TN, FP, TN

Q: Explain how a ROC (Receiver operating characteristic) curve works.

The ROC curve is a graphical representation of the contrast between true positive rates and the false positive rate at various thresholds. It’s often used as a proxy for the trade-off between the sensitivity of the model (true positives) vs specificity (false positives). You want your model to get TPs faster than FPs, if there is the same rate of gettingTP as getting FP your model is useless.

Q: Explain TP/FP/FN/TN in a simple example:

A: - Fire alarm goes off + fire = TP
Fire alarm goes off + no fire = FP
Fire alarm doesn’t go off + fire = FN
Fire alarm doesn’t go off + no fire = TN

Q: What is Bayes Theorem?

A: Essentially Bayes Theorem gives you a probability of an event given what is known as prior knowledge, Prob = TP/ All positives(TP+FP).

Q: Why is “Naive” Bayes naive? 

A: NB makes the naive assumption that the features in a dataset are independent of each other, which isn’t applicable to real-world datasets.

Q: What is a decision tree and when would you choose to use one?

A: As the name suggests decision trees are tree-like model of decisions, they make relations between features easily interpretable. They can be used for both classification (classify passenger as survived or died) and regression (continuous values like price of a house) and don’t require any assumptions of linearity in the data.

Q: How are they pruned?

A: Pruning is what happens in decision trees when branches that have weak predictive power are removed in order to reduce the complexity of the model and increase the predictive accuracy of a decision tree model. Pruning can happen bottom-up and top-down, with approaches such as reduced error pruning and cost complexity pruning.

Reduced error pruning is perhaps the simplest version: replace each node. If it doesn’t decrease predictive accuracy, keep it pruned. While simple, this heuristic actually comes pretty close to an approach that would optimize for maximum accuracy.

Q: What is the difference between Gini Impurity and Entropy in a decision tree? 

A: While both are metrics to decide how to split a tree, Gini measurement is the probability of a random sample being classified correctly by randomly picking a label from the branch. In information theory Entropy is the measured lack of information in a system and you calculate gain by making a split. This delta entropy tells you about how the uncertainty about the label was reduced. Gini is more common because it doesn’t require the log calculations that Entropy takes.

Q: When will Entropy decrease in binary tree classification?

A: It decreases the closer we get to the leaf node.

Q: Why don’t we tend to use linear regression to model binary responses?

A: Linear regression prediction output is continuous, if you want to model binary results you should use logistic regression.

Q: What is the difference between hinge loss and log loss?

A: The hinge loss is used for "maximum-margin" classification, most notably for support vector machines. Logistic loss diverges faster than hinge loss. So, in general, it will be more sensitive to outliers. Hinge loss also penalizes wrong answers, as well as correct unconfident answers.

Q: How do linear and logistic regression differ in their error minimization techniques?

A: Linear regression uses ordinary least squares method to minimize the errors and arrive at a best possible fit, while logistic regression uses maximum likelihood method to arrive at the solution.

Q: What is more important model accuracy or model performance?

A: Model accuracy is actually a subset of model performance. For example, if you wanted to detect fraud in a massive dataset with a sample of millions, a more accurate model would most likely predict no fraud at all if only a vast minority of cases were fraud. However, this would be useless for a predictive model — a model designed to find fraud that asserted there was no fraud at all!

Q: What’s the difference between a generative and discriminative model?

A: Discriminative models are great for classification (SVM, NN, NLPs, facial recognition), they map high dimensional sensory input into a class. A generative models care how the data was generated and will learn will learn categories of data (chatbot, GANs).

Q: How does SVM and logistic regression differ?

A: They only differ in the loss function — SVM minimizes hinge loss while logistic regression minimizes logistic loss.

Q: What is an SVM? What do you do if your data is not linear? (kernel trick)

A: The objective of the support vector machine algorithm is to find the hyperplane that has the maximum margin in an N-dimensional space(N — the number of features) that distinctly classifies the data points. A kernel trick allows you to map your data to a higher dimensional feature space so you can fit a hyperplane. This is done by taking the vectors in the original space and returning the dot product of the vectors in the feature space.

Q: How do you turn the regularization (C) and gamma terms in SVMs?

A: High gamma values mean only data points close to the line are considered and a high C term means a smaller-margin around line (could overfit).

Q: Explain Dijkstra's algorithm? (Know how to use it).

A: Dijkstra's algorithm is an algorithm for finding the shortest paths between nodes in a graph.

Q: How is KNN different from K-means clustering?

A: KNN or K-Nearest Neighbors is a supervised learning method technique used from classification or regression and does not require training. K-means is an unsupervised clustering algorithm fitting to K-clusters. 

Q: What is ensemble learning?

A: Ensemble techniques use a combination of learning algorithms to optimize better predictive performance. And they typically reduce overfitting in models. Ensembling techniques are further classified into Bagging and Boosting.

Q: What is the difference between bagging and boosting?

A: Both are ensemble models that use random sampling to reduce variance. Bagging models are built independently and better solves the problem of overfitting. Boosting builds on top of old models to create models with less bias, also weights the better performing examples higher, but may overfit. 

Q: How do you go from a decision tree to a random forest? To a Gradient Boosted Tree?

A: Bagging takes many uncorrelated learners to make a final model and it reduces error by reducing variance. Example of bagging ensemble are Random Forest models.

Boosting is an ensemble technique in which the predictors are not made independently, but sequentially in order to learn from the mistakes of the previous predictors. Gradient Boosted Trees are an example of boosting algorithm.

Q: Describe a hash table.

A: A hash table is a data structure like a dictionary in python. A key is mapped to certain values through the use of a hash function. They are often used for tasks such as database indexing.

Q: How do you deal with imbalanced data?

A: Collect more data, resample the dataset to correct for imbalances, try difference models or algorithms. 

## ML Validation

Q: Name two ways to evaluate the performance of a classification algorithm.

A: 1) Confusing Matrix ([TN,FP],[FN,TP]) 
     2) Accuracy (also AUC, F1, MAE, MSE)

Q: What’s the difference between Type I and Type II error?

A: Type I error is a false positive, while Type II error is a false negative.

Q: What is the difference between MSE and MAE?

A: MAE loss is more robust to outliers, but its derivatives are not continuous, making it inefficient to find the solution. MSE loss is sensitive to outliers, but gives a more stable and closed form solution (by setting its derivative to 0). Use MAE if you have a lot of anomalies in your dataset.

Q: Why do we need a cost function and which is the best cost to use in classification algorithms.

A: We need a cost function to optimize our weights for model performance and I would use the cost function Mean Squared Error and minimize the MSE to improve the accuracy of our classification model.

Q: How do you ensure you’re not overfitting with a model?

A: 1- Keep the model simpler: reduce variance by taking into account fewer variables and parameters, thereby removing some of the noise in the training data.
2- Use cross-validation techniques such as k-folds cross-validation.
3- Use regularization techniques such as LASSO that penalize certain model parameters if they’re likely to cause overfitting.

Q: Explain the cross-validation resampling procedure.

A: The general procedure is as follows:

1) Shuffle the dataset randomly.
2) Split the dataset into k groups
3) For each unique group:
Take the group as a hold out or test data set
Take the remaining groups as a training data set
Fit a model on the training set and evaluate it on the test set
Retain the evaluation score and discard the model
4)Combine evaluation scores into single average (CV error)
5)Repeat process for different model and choose the one w/ lowest CV error

Q: How does evaluating your model differ between using CV or bootstrapping? What is MC-CV?

A: CV tends to be less biased but K-fold CV has fairly large variance. On the other hand, bootstrapping (sampling with replacement) tends to drastically reduce the variance but gives more biased results (they tend to be pessimistic). "Monte Carlo CV" aka "leave-group-out CV" does many random splits of the data to reduce variance.

Q: What’s the trade-off between bias and variance?

A: Bias is due to overly simplistic assumptions while variance is error due to too much complexity in the learning algorithm you’re using. Bias leads to under-fitting your data and variance leads to overfitting your data. Essentially, if you make the model more complex and add more variables, you’ll lose bias but gain some variance — in order to get the optimally reduced amount of error, you’ll have to tradeoff bias and variance and try your best to minimize each.
(In machine learning/statistics as a whole, accuracy vs. precision is analogous to bias vs. variance).

Q: What cross-validation technique would you use on a time series dataset?

A: Instead of using standard k-folds cross-validation, you have to pay attention to the fact that a time series is not randomly distributed data — it is inherently ordered by chronological order. If a pattern emerges in later time periods for example, your model may still pick up on it even if that effect doesn’t hold in earlier years!

You’ll want to do something like forward chaining where you’ll be able to model on past data then look at forward-facing data.

fold 1 : training [1], test [2]
fold 2 : training [1 2], test [3]
fold 3 : training [1 2 3], test [4]
fold 4 : training [1 2 3 4], test [5]
fold 5 : training [1 2 3 4 5], test [6]

Q: What’s the difference between L1 and L2 regularization? How does it solve the problem of overfitting? Which regularizer to use and when?

A: When dealing with a large number of features we no longer want to use CV. Both L1 (Lasso Regression) and L2 (Ridge Regression) regularization techniques are used to address over-fitting and feature selection, the key difference between these two is the penalty term. Lasso Regression (Least Absolute Shrinkage and Selection Operator) adds “absolute value of magnitude” of coefficient while Ridge regression adds “squared magnitude” of coefficient as penalty term to the loss function. 
The key difference between these techniques is that Lasso is more binary/sparse and shrinks the less important feature’s coefficient to zero thus, removing some feature altogether and L2 regularization tends to spread error among all the term. L1 works well for feature selection in case we have a huge number of features.

## ML Stats

Q: What is a Fourier transform? And why do we use it.

A:  Given a smoothie, it’s how we find the recipe (in terms of superposition of symmetric functions). Fourier transforms are used to it’s a extract features from audio signals by converting a signal from time to frequency domain.

Q: What’s the difference between probability and likelihood?

A: For binomial distributions: Probability is the percentage that a success occur. Likelihood is the conditional probability, i.e. the probability that the above event will happen.

Q: What is the difference between PCA and t-SNE? What are their use cases?

A: Both methods are used for dimensionality reduction, but t-SNE tries to deconvolve relationships between neighbors in high-dimensional data to understand the underlying structure of the data. Principal component analysis first identifies the hyperplane that lies closest to the data, and then projects the data onto it. PCA preserves the maximum amount of variance and requires labels, but is much less computationally expensive than t-SNE.

Q: How do eigenvalues and eigenvectors relate to PCA?

A: Eigenvectors have corresponding eigenvalues and eigenvectors that have the largest eigenvalues will be the principal components (new dimensions of our data).

Q: What is Maximum Likelihood (MLE)?

A: Maximum likelihood estimation is a method that determines values for the parameters of a model. The parameter values are found such that they maximize the likelihood that the process described by the model produced the data that were actually observed.

Q: When are Maximum Likelihood and Least Squared Error equal?

A: For least squares parameter estimation we want to find the line that minimizes the total squared distance between the data points and the regression line. In maximum likelihood estimation we want to maximize the total probability of the data. When a Gaussian distribution is assumed, the maximum probability is found when the data points get closer to the mean value. Since the Gaussian distribution is symmetric, this is equivalent to minimizing the distance between the data points and the mean value.

### Sources: 
Astronomer Amber!
https://ml-cheatsheet.readthedocs.io/en/latest/forwardpropagation.html
Machine Learning Interview Questions and Answers | Machine Learning Interview Preparation | Edureka
https://www.springboard.com/blog/machine-learning-interview-questions/
https://machinelearningmastery.com/k-fold-cross-validation/
https://towardsdatascience.com/l1-and-l2-regularization-methods-ce25e7fc831c
https://towardsdatascience.com/support-vector-machine-vs-logistic-regression-94cc2975433f
https://towardsdatascience.com/probability-concepts-explained-maximum-likelihood-estimation-c7b4342fdbb1
https://medium.com/mlreview/gradient-boosting-from-scratch-1e317ae4587d


# Solutions on Cracking The Machine Learning Interview
### -------> Currently under construction! <-------
I am currently writing a solution from the Medium article "Cracking the Machine Learning Interview," written by Subhrajit Roy. In the past year since the article went public, Subhrajit has only written down the questions with no update on the solutions. I plan on finishing the war. I may add more questions outside of the articles domain. Some of my solutions may contain code from the following following frameworks: Scikit-Learn, PyTorch, TensorFlow

https://medium.com/subhrajit-roy/cracking-the-machine-learning-interview-1d8c5bb752d8

<a href="https://medium.com/subhrajit-roy/cracking-the-machine-learning-interview-1d8c5bb752d8" target="_blank">
  <img src="https://github.com/rchavezj/Cracking_The_Machine_Learning_Interview/blob/master/crackingTheMachineLearningInterviewCover.png">
</a>

# Contents: 
|                        |                                          |
| ---------------------- | ---------------------------------------- |
| 1. [Linear Algebra](#Linear-Algebra)                         | 2. [Numerical Optimization](#Numerical-Optimization)                                         |
| 3. [Basics of Probability and Information Theory](#Basics-of-Probability-and-Information-Theory)                                                                                                        |  4. [Confidence Interval](#Confidence-Interval)|
| 5. [Learning Theory](#Learning-Theory)                       |  6. [Model and Feature Selection](#Model-and-Feature-Selection) |
| 7. [Curse of dimensionality](#Curse-of-Dimensionality)       |  8. [Universal approximation of neural networks](#Universal-Approximation-of-Neural-Networks) |
| 9. [Deep Learning motivation](#Deep-Learning-Motivation)     |  10. [Support Vector Machine](#Support-Vector-Machine) |
| 11. [Bayesian Machine Learning](#Bayesian-Machine-Learning)  |  12. [Regularization](#Regularization) |
| 13. [Evaluation of Machine Learning systems](#Evaluation-of-Machine-Learning-Systems) |  14. [Clustering](#Clustering)  |
| 15. [Dimensionality Reduction](#Dimensionality-Reduction)    |  16. [Basics of Natural Language Processing](#Basics-of-Natural-Language-Processing) |
| 17. [Some basic questions](#Some-basic-questions)            |  18. [Optimization Procedures](#Optimization-Procedures) |
| 19. [Sequence Modeling](#Sequence-Modeling)                  |  20. [Autoencoders](#Autoencoders)               |
| 21. [Representation Learning](#Representation-Learning)      |  22. [Monte Carlo Methods](#Monte-Carlo-Methods) |
| 23. [Generative Models](#Generative-Models)                  |  24. [Reinforcement Learning](#Reinforcement-Learning) |
| 25. [Probabilistic Graphical Models](#Probabilistic-Graphical-Models)   | 26. [Computational Logic](#Computational-Logic)     |



### [Linear Algebra](01_Linear_Algebra/#Linear-Algebra)
[(Return back to Contents)](#Contents)
<img src="01_Linear_Algebra/linear_algebra.png" width="700">

1. What is broadcasting in connection to Linear Algebra?
2. [What are scalars, vectors, matrices, and tensors?](01_Linear_Algebra/#2-What-are-scalars-vectors-matrices-and-tensors)
3. What is Hadamard product of two matrices?
4. What is an inverse matrix?
5. If inverse of a matrix exists, how to calculate it?
6. What is the determinant of a square matrix? How is it calculated? What is the connection of determinant to eigenvalues?
7. Why does the negative area of the determinent relate to orientation flipping? Check out lecture 6 from 3BLUE1BROWN? 
8. Justify in one sentence why the following equation on why it is true: "If you multiply two matrices together, the determinent of the reulting matrix is the same as the product of the determinence of the original two matrices" det(M_{1}M_{2}) = det(M_{1})det(M_{2}). If you try to justify with numbers it would take a long time. 
9.  Discuss span and linear dependence.
10. Following up on question #7, what does the following definition mean, "The basis of a vector space is a set of linearly independent vectors that span the full space."
11. What is Ax = b? When does Ax = b has a unique solution?
12. In Ax = b, what happens when A is fat or tall?
11. When does inverse of A exist?
12. [What is a norm? What is L1, L2 and L infinity norm?](#)
13. What are the conditions a norm has to satisfy?
14. Why is squared of L2 norm preferred in ML than just L2 norm?
15. When L1 norm is preferred over L2 norm?
16. Can the number of nonzero elements in a vector be defined as L0 norm? If no, why?
17. What is Frobenius norm?
18. What is a diagonal matrix?
19. Why is multiplication by diagonal matrix computationally cheap? How is the multiplication different for square vs. non-square diagonal matrix?
20. At what conditions does the inverse of a diagonal matrix exist?
21. What is a symmetrix matrix?
22. What is a unit vector?
23. When are two vectors x and y orthogonal?
24. At R^n what is the maximum possible number of orthogonal vectors with non-zero norm?
25. When are two vectors x and y orthonormal?
26. What is an orthogonal matrix? Why is computationally preferred?
27. What is eigendecomposition, eigenvectors and eigenvalues?
28. How to find eigen values of a matrix?
29. Write the eigendecomposition formula for a matrix. If the matrix is real symmetric, how will this change?
30. Is the Eigendecomposition guaranteed to be unique? If not, then how do we represent it?
31. What are positive definite, negative definite, positive semi definite and negative semi definite matrices?
32. What is Singular Value Decomposition? Why do we use it? Why not just use ED?
33. Given a matrix A, how will you calculate its Singular Value Decomposition?
34. What are singular values, left singulars and right singulars?
35. What is the connection of Singular Value Decomposition of A with functions of A?
36. Why are singular values always non-negative?
37. What is the Moore Penrose pseudo inverse and how to calculate it?
38. If we do Moore Penrose pseudo inverse on Ax = b, what solution is provided is A is fat? Moreover, what solution is provided if A is tall?
39. Which matrices can be decomposed by ED?
40. Which matrices can be decomposed by SVD?
41. What is the trace of a matrix?
42. How to write Frobenius norm of a matrix A in terms of trace?
43. Why is trace of a multiplication of matrices invariant to cyclic permutations?
44. What is the trace of a scalar?
45. Write the frobenius norm of a matrix in terms of trace?



### Numerical Optimization
[(Return back to Contents)](#Contents)

<img src="02_Numerical_Optimization/optimization_cover.png">

1. What is underflow and overflow?
2. How to tackle the problem of underflow or overflow for softmax function or log softmax function?
3. What is poor conditioning?
4. What is the condition number?
5. What are grad, div and curl?
6. What are critical or stationary points in multi-dimensions?
7. Why should you do gradient descent when you want to minimize a function?
8. What is line search?
9. What is hill climbing?
10. What is a Jacobian matrix?
11. What is curvature?
12. What is a Hessian matrix?
13. What is a gradient checking?



### Basics of Probability and Information Theory
[(Return back to Contents)](#Contents)

<img src="03_Basics_of_Probability_and_Information_Theory/Basics_of_Probability_and_Information_Theory.png">

1. Compare “Frequentist probability” vs. “Bayesian probability”?
2. What is a random variable?
3. What is a probability distribution?
4. What is a probability mass function?
5. What is a probability density function?
6. What is a joint probability distribution?
7. What are the conditions for a function to be a probability mass function?
8. What are the conditions for a function to be a probability density function?
9. What is a marginal probability? Given the joint probability function, how will you calculate it?
10. What is conditional probability? Given the joint probability function, how will you calculate it?
11. State the Chain rule of conditional probabilities.
12. What are the conditions for independence and conditional independence of two random variables?
13. What are expectation, variance and covariance?
14. Compare covariance and independence.
15. What is the covariance for a vector of random variables?
16. What is a Bernoulli distribution? Calculate the expectation and variance of a random variable that follows Bernoulli distribution?
17. What is a multinoulli distribution?
18. What is a normal distribution?
19. Why is the normal distribution a default choice for a prior over a set of real numbers?
20. What is the central limit theorem?
21. What are exponential and Laplace distribution?
22. What are Dirac distribution and Empirical distribution?
23. What is mixture of distributions?
24. Name two common examples of mixture of distributions? (Empirical and Gaussian Mixture)
25. Is Gaussian mixture model a universal approximator of densities?
26. Write the formulae for logistic and softplus function.
27. Write the formulae for Bayes rule.
28. What do you mean by measure zero and almost everywhere?
29. If two random variables are related in a deterministic way, how are the PDFs related?
30. Define self-information. What are its units?
31. What are Shannon entropy and differential entropy?
32. What is Kullback-Leibler (KL) divergence?
33. Can KL divergence be used as a distance measure?
34. Define cross-entropy.
35. What are structured probabilistic models or graphical models?
36. In the context of structured probabilistic models, what are directed and undirected models? How are they represented? What are cliques in undirected structured probabilistic models?


### Confidence interval 
[(Return back to Contents)](#Contents)

<img src="04_Confidence_Interval/04_Confidence_Interval.png">

1. What is population mean and sample mean?
2. What is population standard deviation and sample standard deviation?
3. Why population s.d. has N degrees of freedom while sample s.d. has N-1 degrees of freedom? In other words, why 1/N inside root for pop. s.d. and 1/(N-1) inside root for sample s.d.?
4. What is the formula for calculating the s.d. of the sample mean?
5. What is confidence interval?
6. What is standard error?



### Learning Theory 
[(Return back to Contents)](#Contents)

<img src="05_Learning_Theory/ml_learning_theory.png">

1. Describe bias and variance with examples.
2. What is Empirical Risk Minimization?
3. What is Union bound and Hoeffding’s inequality?
4. Write the formulae for training error and generalization error. Point out the differences.
5. State the uniform convergence theorem and derive it.
6. What is sample complexity bound of uniform convergence theorem?
7. What is error bound of uniform convergence theorem?
8. What is the bias-variance trade-off theorem?
9. From the bias-variance trade-off, can you derive the bound on training set size?
10. What is the VC dimension?
11. What does the training set size depend on for a finite and infinite hypothesis set? Compare and contrast.
12. What is the VC dimension for an n-dimensional linear classifier?
13. How is the VC dimension of a SVM bounded although it is projected to an infinite dimension?
14. Considering that Empirical Risk Minimization is a NP-hard problem, how does logistic regression and SVM loss work?



### Model and feature selection
[(Return back to Contents)](#Contents)

<img src="06_Feature_Engineering/06_Feature_Engineering.png">

1. Why are model selection methods needed?
2. How do you do a trade-off between bias and variance?
3. What are the different attributes that can be selected by model selection methods?
4. Why is cross-validation required?
5. Describe different cross-validation techniques.
6. What is hold-out cross validation? What are its advantages and disadvantages?
7. What is k-fold cross validation? What are its advantages and disadvantages?
8. What is leave-one-out cross validation? What are its advantages and disadvantages?
9. Why is feature selection required?
10. Describe some feature selection methods.
11. What is forward feature selection method? What are its advantages and disadvantages?
12. What is backward feature selection method? What are its advantages and disadvantages?
13. What is filter feature selection method and describe two of them?
14. What is mutual information and KL divergence?
15. Describe KL divergence intuitively.



### Curse of dimensionality
[(Return back to Contents)](#Contents)

<img src="07_Curse_Of_Dimensionality/Curse_Of_Dimensionality.jpeg">

1. Describe the curse of dimensionality with examples.
2. What is local constancy or smoothness prior or regularization?



### Universal approximation of neural networks
[(Return back to Contents)](#Contents)

<img src="08_Universal_Approximation_of_Neural_Networks/08_Universal_Approximation_of_Neural_Networks.png">

1. State the universal approximation theorem? What is the technique used to prove that?
2. What is a Borel measurable function?
3. Given the universal approximation theorem, why can’t a Multi Layer Perceptron (MLP) still reach an arbitrarily small positive error?



### Deep Learning motivation
[(Return back to Contents)](#Contents)

<img src="09_Deep_Learning_Motivation/09_Deep_Learning_Motivation.jpg">

1. What is the mathematical motivation of Deep Learning as opposed to standard Machine Learning techniques?
2. In standard Machine Learning vs. Deep Learning, how is the order of number of samples related to the order of regions that can be 3. recognized in the function space?
3. What are the reasons for choosing a deep model as opposed to shallow model?
4. How Deep Learning tackles the curse of dimensionality?



### Support Vector Machine
[(Return back to Contents)](#Contents)

<img src="10_Support_Vector_Machine/10_Support_Vector_Machine.png">

1. How can the SVM optimization function be derived from the logistic regression optimization function?
2. What is a large margin classifier?
3. Why SVM is an example of a large margin classifier?
4. SVM being a large margin classifier, is it influenced by outliers?
5. What is the role of C in SVM?
6. In SVM, what is the angle between the decision boundary and theta?
7. What is the mathematical intuition of a large margin classifier?
8. What is a kernel in SVM? Why do we use kernels in SVM?
9. What is a similarity function in SVM? Why it is named so?
10. How are the landmarks initially chosen in an SVM? How many and where?
11. Can we apply the kernel trick to logistic regression? Why is it not used in practice then?
12. What is the difference between logistic regression and SVM without a kernel?
13. How does the SVM parameter C affect the bias/variance trade off?
14. How does the SVM kernel parameter sigma² affect the bias/variance trade off?
15. Can any similarity function be used for SVM?
16. Logistic regression vs. SVMs: When to use which one?



### Bayesian Machine Learning
[(Return back to Contents)](#Contents)

<img src="11_Bayesian_Machine_Learning/11_Bayesian_Machine_Learning.jpg">

1. What are the differences between “Bayesian” and “Freqentist” approach for Machine Learning?
2. Compare and contrast maximum likelihood and maximum a posteriori estimation.
3. How does Bayesian methods do automatic feature selection?
4. What do you mean by Bayesian regularization?
5. When will you use Bayesian methods instead of Frequentist methods?
6. Please explain Expectation-Maximization algorithm
7. What is Variational Inference?
8. What is Latent Dirichlet Allocation (LDA)?
9. What is Markov chain?



### Regularization
[(Return back to Contents)](#Contents)

<img src="12_Regularization/12_Regularization.png">

1. What is L1 regularization?
2. What is L2 regularization?
3. Compare L1 and L2 regularization.
4. Why does L1 regularization result in sparse models?
5. What is dropout?
6. How will you implement dropout during forward and backward pass?



### Evaluation of Machine Learning systems
[(Return back to Contents)](#Contents)

<img src="13_Evaluation_of_Machine_Learning_Systems/13_Evaluation_of_Machine_Learning_Systems.jpg">

1. What are accuracy, sensitivity, specificity, ROC?
2. What are precision and recall?
3. Describe t-test in the context of Machine Learning.



### Clustering
[(Return back to Contents)](#Contents)

<img src="14_Clustering/14_Clustering.png">

1. Describe the k-means algorithm.
2. What is distortion function? Is it convex or non-convex?
3. Tell me about the convergence of the distortion function.
4. Topic: EM algorithm
5. What is the Gaussian Mixture Model?
6. Describe the EM algorithm intuitively.
7. What are the two steps of the EM algorithm
8. Compare Gaussian Mixture Model and Gaussian Discriminant Analysis.



### Dimensionality Reduction
[(Return back to Contents)](#Contents)

<img src="15_Dimensionality_Reduction/15_Dimensionality_Reduction.png">

1. Why do we need dimensionality reduction techniques?
2. What do we need PCA and what does it do?
3. What is the difference between logistic regression and PCA?
4. What are the two pre-processing steps that should be applied before doing PCA?



### Basics of Natural Language Processing
[(Return back to Contents)](#Contents)

<img src="16_Basics_of_Natural_Language_Processing/16_Basics_of_Natural_Language_Processing.png">

1. What is WORD2VEC?
2. What is t-SNE? Why do we use PCA instead of t-SNE?
3. What is sampled softmax?
4. Why is it difficult to train a RNN with SGD?
5. How do you tackle the problem of exploding gradients?
6. What is the problem of vanishing gradients?
7. How do you tackle the problem of vanishing gradients?
8. Explain the memory cell of a LSTM.
9. What type of regularization do one use in LSTM?
10. What is Beam Search?
11. How to automatically caption an image?



### Some basic questions
[(Return back to Contents)](#Contents)

<img src="17_Some_basic_Questions/17_Some_basic_Questions.png">

1. Can you state Tom Mitchell’s definition of learning and discuss T, P and E?
2. What can be different types of tasks encountered in Machine Learning?
3. What are supervised, unsupervised, semi-supervised, self-supervised, multi-instance learning, and reinforcement learning?
4. Loosely how can supervised learning be converted into unsupervised learning and vice-versa?
5. Consider linear regression. What are T, P and E?
6. Derive the normal equation for linear regression.
7. What do you mean by affine transformation? Discuss affine vs. linear transformation.
8. Discuss training error, test error, generalization error, overfitting, and underfitting.
9. Compare representational capacity vs. effective capacity of a model.
Discuss VC dimension.
10. What are nonparametric models? What is nonparametric learning?
11. What is an ideal model? What is Bayes error? What is/are the source(s) of Bayes error occur?
12. What is the no free lunch theorem in connection to Machine Learning?
13. What is regularization? Intuitively, what does regularization do during the optimization procedure?
14. What is weight decay? What is it added?
15. What is a hyperparameter? How do you choose which settings are going to be hyperparameters and which are going to be learned?
16. Why is a validation set necessary?
17. What are the different types of cross-validation? When do you use which one?
18. What are point estimation and function estimation in the context of Machine Learning? What is the relation between them?
19. What is the maximal likelihood of a parameter vector $theta$? Where does the log come from?
20. Prove that for linear regression MSE can be derived from maximal likelihood by proper assumptions.
21. Why is maximal likelihood the preferred estimator in ML?
22. Under what conditions do the maximal likelihood estimator guarantee consistency?
23. What is cross-entropy of loss?
24. What is the difference between loss function, cost function and objective function?



### Optimization procedures
[(Return back to Contents)](#Contents)

<img src="18_Optimization_Procedures/18_Optimization_Procedures.png">

1. What is the difference between an optimization problem and a Machine Learning problem?
2. How can a learning problem be converted into an optimization problem?
3. What is empirical risk minimization? Why the term empirical? Why do we rarely use it in the context of deep learning?
4. Name some typical loss functions used for regression. Compare and contrast.
5. What is the 0–1 loss function? Why can’t the 0–1 loss function or classification error be used as a loss function for optimizing a deep neural network?



### Sequence Modeling
[(Return back to Contents)](#Contents)

<img src="19_Sequence_Modeling/19_Sequence_Modeling.jpg">

1. Write the equation describing a dynamical system. Can you unfold it? Now, can you use this to describe a RNN?
2. What determines the size of an unfolded graph?
3. What are the advantages of an unfolded graph?
4. What does the output of the hidden layer of a RNN at any arbitrary time t represent?
5. Are the output of hidden layers of RNNs lossless? If not, why?
6. RNNs are used for various tasks. From a RNNs point of view, what tasks are more demanding than others?
7. Discuss some examples of important design patterns of classical RNNs.
8. Write the equations for a classical RNN where hidden layer has recurrence. How would you define the loss in this case? What problems you might face while training it?
9. What is backpropagation through time?
10. Consider a RNN that has only output to hidden layer recurrence. What are its advantages or disadvantages compared to a RNN having only hidden to hidden recurrence?
11. What is Teacher forcing? Compare and contrast with BPTT.
12. What is the disadvantage of using a strict teacher forcing technique? How to solve this?
13. Explain the vanishing/exploding gradient phenomenon for recurrent neural networks.
14. Why don’t we see the vanishing/exploding gradient phenomenon in feedforward networks?
15. What is the key difference in architecture of LSTMs/GRUs compared to traditional RNNs?
16. What is the difference between LSTM and GRU?
17. Explain Gradient Clipping.
18. Adam and RMSProp adjust the size of gradients based on previously seen gradients. Do they inherently perform gradient clipping? If no, why?
19. Discuss RNNs in the context of Bayesian Machine Learning.
20. Can we do Batch Normalization in RNNs? If not, what is the alternative?



### Autoencoders
[(Return back to Contents)](#Contents)

<img src="20_Autoencoders/20_Autoencoders.png">

1. What is an Autoencoder? What does it “auto-encode”?
2. What were Autoencoders traditionally used for? Why there has been a resurgence of Autoencoders for generative modeling?
3. What is recirculation?
4. What loss functions are used for Autoencoders?
5. What is a linear autoencoder? Can it be optimal (lowest training reconstruction error)? If yes, under what conditions?
6. What is the difference between Autoencoders and PCA?
7. What is the impact of the size of the hidden layer in Autoencoders?
8. What is an undercomplete Autoencoder? Why is it typically used for?
9. What is a linear Autoencoder? Discuss it’s equivalence with PCA. Which one is better in reconstruction?
10. What problems might a nonlinear undercomplete Autoencoder face?
11. What are overcomplete Autoencoders? What problems might they face? Does the scenario change for linear overcomplete autoencoders?
12. Discuss the importance of regularization in the context of Autoencoders.
13. Why does generative autoencoders not require regularization?
14. What are sparse autoencoders?
15. What is a denoising autoencoder? What are its advantages? How does it solve the overcomplete problem?
16. What is score matching? Discuss it’s connections to DAEs.
17. Are there any connections between Autoencoders and RBMs?
18. What is manifold learning? How are denoising and contractive autoencoders equipped to do manifold learning?
19. What is a contractive autoencoder? Discuss its advantages. How does it solve the overcomplete problem?
20. Why is a contractive autoencoder named so?
21. What are the practical issues with CAEs? How to tackle them?
22. What is a stacked autoencoder? What is a deep autoencoder? Compare and contrast.
23. Compare the reconstruction quality of a deep autoencoder vs. PCA.
24. What is predictive sparse decomposition?
25. Discuss some applications of Autoencoders.



### Representation Learning
[(Return back to Contents)](#Contents)

<img src="21_Representation_Learning/21_Representation_Learning.png">

1. What is representation learning? Why is it useful?
2. What is the relation between Representation Learning and Deep Learning?
3. What is one-shot and zero-shot learning (Google’s NMT)? Give examples.
4. What trade offs does representation learning have to consider?
5. What is greedy layer-wise unsupervised pretraining (GLUP)? Why greedy? Why layer-wise? Why unsupervised? Why pretraining?
6. What were/are the purposes of the above technique? (deep learning problem and initialization)
7. Why does unsupervised pretraining work?
8. When does unsupervised training work? Under which circumstances?
9. Why might unsupervised pretraining act as a regularizer?
10. What is the disadvantage of unsupervised pretraining compared to other forms of unsupervised learning?
11. How do you control the regularizing effect of unsupervised pretraining?
12. How to select the hyperparameters of each stage of GLUP?



### Monte Carlo Methods
[(Return back to Contents)](#Contents)

<img src="22_Monte_Carlo_Methods/22_Monte_Carlo_Methods.png">

1. What are deterministic algorithms?
2. What are Las vegas algorithms?
3. What are deterministic approximate algorithms?
4. What are Monte Carlo algorithms?



### Generative Models
[(Return back to Contents)](#Contents)

<img src="23_Generative_Models/23_Generative_Models.png">

1. What is a Variational Autoencoder (VAE)?
2. How is VAE different from a regular Autoencoder?
3. Basics of GAN?
4. How do you train a GAN (Backpropagation)?
5. Cost function derivation?
6. What are the drawbacks for GAN?
7. Implement GAN with PyTorch
8. Implement GAN with Tensorflow



### Reinforcement Learning
[(Return back to Contents)](#Contents)

<img src="24_Reinforcement_Learning/24_Reinforcement_Learning.png">

1. What is the Reinforcement Learning?
2. Factors in Reinforcement Learning with Python
3. Types of Reinforcement Learning with Python
4. Positive Reinforcement Learning
5. Negative Reinforcement Learning
6. Reinforced Learning vs Supervised Learning
7. Decision Making
8. Dependency and Labels




Every resource needs to be broken down and put into separate sections (and cite where they came from).

    https://towardsdatascience.com/how-to-ace-data-science-interviews-statistics-f3d363ad47b

    Algorithms used on daily basis by data scientist: https://www.kdnuggets.com/2018/04/key-algorithms-statistical-models-aspiring-data-scientists.html

    http://houseofbots.com/news-detail/2851-4-this-is-what-i-really-do-as-a-data-scientist

    https://medium.com/open-machine-learning-course/open-machine-learning-course-topic-7-unsupervised-learning-pca-and-clustering-db7879568417

    https://www.udemy.com/python-for-data-structures-algorithms-and-interviews/learn/v4/overview

    http://nbviewer.jupyter.org/github/jmportilla/Python-for-Algorithms--Data-Structures--and-Interviews/tree/master/

    https://towardsdatascience.com/data-science-and-machine-learning-interview-questions-3f6207cf040b

    useful: http://houseofbots.com/news-detail/2248-4-109-commonly-asked-data-science-interview-questions

    https://medium.com/acing-ai/google-ai-interview-questions-acing-the-ai-interview-1791ad7dc3ae

    https://medium.com/acing-ai

#### Questions on Deep Learning

* Design a network to detect two object classes if you know there is going to be only single instance of each object in the image. How the design changes if multiple, unknown number of instances are present? How the design and strategy changes if the number of object classes to be detected is huge ( > 10K)?
* Let me also share questions from published material which tests if the candidate is well prepared to understand the current literature/existing approaches. This is by no means an exhaustive list:
* Semantic segmentation, Object detection
* Explain max un-pooling operation for increasing the resolution of feature maps.
* What is a Learnable up-sampling or Transpose convolution ?
* Describe the transition between R-CNN, Fast R-CNN and Faster RCNN for object detection.
* Describe how RPNs are trained for prediction of region proposals in Faster R-CNN?
* Describe the approach in SSD and YOLO for object detection. How these approaches differ from Faster-RCNN. When will you use one over the other?
* Difference between Inception v3 and v4. How does Inception Resnet compare with V4.
* Explain main ideas behind ResNet? Why would you try ResNet over other architectures?
* Explain batch gradient descent, stochastic gradient descent and mini-match gradient descent.
* Loss functions: Cross-entropy, L2, L1
* Explain Dropout and Batch Normalization. Why Batch Normalization helps in faster convergence?
* Are neural networks and deep learning overrated?
* Can deep learning and neural networks be patented?
* What leadership questions should I expect from an Amazon on-site interview for a Software Engineering role?
* What is learning in neural network?
* What is the difference between Neural Networks and Deep Learning?
* Interestingly, this question as applied to Deep Learning does have a definitive answer for me, whereas the general form of the question may not.
* It is a relatively new topic in the general software engineering population. It has not yet been taught for years in college by professors who have extracted insightful ways to teach the fundamentals. So a lot knowledge here is gleaned from watching advanced talks and reading research papers. Unfortunately, this also means that many candidates have a strong functional knowledge of the state-of-the-art Whats and Hows, yet not fully mastering the Whys.
* So, I find that there are indeed "toughest NN and Deep Learning" questions, where many otherwise knowledgeable candidates fall down. They might give you technically correct answers, using lots of jargon, but never getting to the heart of the issue. They might give you an answer involving a lot of correct Hows, but that reveal they don't really understand the fundamental Whys. The best answers to these questions cannot (yet) be easily Googled. They are invariably of this pattern:
* Explain the following, so that a colleague new to the field/an eighth grader can understand (in no particular order, not exhaustive):
* What is an auto-encoder? Why do we "auto-encode"? Hint: it's really a misnomer.
* What is a Boltzmann Machine? Why a Boltzmann Machine?
* Why do we use sigmoid for an output function? Why tanh? Why not cosine? Why any function in particular?
* Why are CNNs used primarily in imaging and not so much other tasks?
* Explain backpropagation. Seriously. To the target audience described above.
* Is it OK to connect from a Layer 4 output back to a Layer 2 input?
* A data-scientist person recently put up a YouTube video explaining that the essential difference between a Neural Network and a Deep Learning network is that the former is trained from output back to input, while the latter is trained from input toward output. Do you agree? Explain.

* Try these yourself and see if you do indeed have mastery of the fundamentals. If you do, an eighth grader ought to be able to understand and repeat your explanation.
*
* I had some “deep learning interviews” recently, and I thought I could share some questions. First of all, be aware that most of the time, questions don’t have a single answer, and the interviewer just wants to talk with you to see if you are confident about the notions.
* Usually the first questions are : what do you know about some “pre-deep learning epoch” algorithms, like SVM, KNN, Kmeans, Random Forest…?
* Talking about deep learning, here are the questions I was asked to answer:
* Implement dropout during forward and backward pass?
* Was not very hard, you just have to consider what’s happening during testing vs training phase. In this question, the interviewer can test your knowledge on dropout, and backprop
* Neural network training loss/testing loss stays constant, what do you do?
* Open question (ask if there could be an error in your code, going deeper, going simpler…)
* Why do RNNs have a tendency to suffer from exploding/vanishing gradient?
* And probably you know the next question… How to prevent this? You can talk about LSTM cell which helps the gradient from vanishing, but make sure you know why it does so. I also remember having a nice conversation about gradient clipping, where we wonder whether we should clip the gradient element wise, or clip the norm of the gradient.
* Then I had a lot of question about some modern architecture, such as Do you know GAN, VAE, and memory augmented neural network? Can you talk about it?
* Of course, let me talk about the beauty of variational auto encoder.
* Some maths questions such as: Does using full batch means that the convergence is always better given unlimited power?
* What is the problem with sigmoid during backpropagation?
* Very small, between 0.25 and zero.[2]
* Given a black box machine learning algorithm that you can’t modify, how could you improve its error?
* Open question, you can transform the input for example.
* How to find the best hyper parameters?
* Random search, grid search, Bayesian search (and what it is?)
* What is transfer learning?
* I was also asked to implement some papers idea, but it was more as an assignment, than during an interview. Finally I also get non ML questions, more like algorithmic questions


* Good luck for your interview. If you are enough curious, and have a correct knowledge of the field, they will notice it, and you will pass a good moment with the interviewer.
*
* However, I do have some questions to test whether candidates really understand deep learning.
* Can they derive the back-propagation and weights update?
* Extend the above question to non-trivial layers such as convolutional layers, pooling layers, etc.
* How to implement dropout
* Their intuition when and why some tricks such as max pooling, ReLU, maxout, etc. work. There are no right answers but it helps to understand their thoughts and research experience.
* Can they abstract the forward, backward, update operations as matrix operations, to leverage BLAS and GPU?
* If a candidate shows early signs that he/she is an expert in DL, it's not necessary to ask all those questions in details. We can discuss one of their papers or a recent hot paper or something that is not necessarily DL.
* 86.3k Views · View 220 Upvoters
* Related Questions
* What are the learning algorithm in deep neural network?
* How can a neural network learn itself?
* Do you have to show your face during a Google Hangouts interview?
* What is neural networking?
* How can deep learning networks generate images?
* How do neural networks of neural networks behave?
* What is the best book or resource to learn about Neural Networks and Deep Neural Networks?
* What is the best YouTube channel to learn deep learning and neural networks?
* What topics come under deep learning other than neural networks?
* How do I implement deep neural network?
* What is the difference between neural networks and deep neural networks?
* Why are deep networks characterized by neural networks?
* ELI5: What are neural networks?
* What are neural networks in machine learning?
* What are the deep learning algorithms other than neural networks?
* Are neural networks and deep learning overrated?
* Can deep learning and neural networks be patented?
* What leadership questions should I expect from an Amazon on-site interview for a Software Engineering role?
* What is learning in neural network?
* What is the difference between Neural Networks and Deep Learning?
* What are the learning algorithm in deep neural network?
* How can a neural network learn itself?
* Do you have to show your face during a Google Hangouts interview?
* What is neural networking?
* How can deep learning networks generate images?
'
source    
[1]  https://www.quora.com/What-are-the-toughest-neural-networks-and-deep-learning-interview-questions'
[2] Is full-batch gradient descent, with unlimited computer power, always better than mini-batch gradient descent?

# ML-Interview
This is a list of resources I found useful during my preparation for interviews. Broadly speaking, I interviewd for three different profiles: Machine Learning Engineer, Applied Scientist and Quantitative Researcher. 

NOTE: All these profiles usually include multiple "traditional" programming/algorithm rounds, and for that, I relied upon mild leetcoding spree, spread over a period of 3 months. 

## Classical Machine Learning

+ A very good (slighly advanced) course on Machine Learning by Alex Smola. [Link](http://alex.smola.org/teaching/cmu2013-10-701/stats.html) 
+ Perhaps everything that you'll ever need to know for the interview sake. [Link](http://alumni.media.mit.edu/~tpminka/statlearn/glossary/) 
+ Generative vs Discriminative Classifiers (you should know the difference, and tradeoffs when choosing one over the other) [Link](http://www.cs.cmu.edu/~tom/mlbook/NBayesLogReg.pdf)
+ Gradient Boosted Trees [Link](https://web.njit.edu/~usman/courses/cs675_spring20/BoostedTree.pdf)
+ Gentle Introduction to Gradient Boosting [Link](https://www.ccs.neu.edu/home/vip/teach/MLcourse/4_boosting/slides/gradient_boosting.pdf)
+ ROC and AUC (I like this video) [Link](https://www.youtube.com/watch?v=OAl6eAyP-yo&t=729s)
+ Clustering (from Ryan Tibshirani's Data Mining course, other slides are really good as well) [Link 1](https://www.stat.cmu.edu/~ryantibs/datamining/lectures/04-clus1.pdf) [Link 2](https://www.stat.cmu.edu/~ryantibs/datamining/lectures/05-clus2.pdf) [Link 3](https://www.stat.cmu.edu/~ryantibs/datamining/lectures/06-clus3.pdf)
+ Good old Linear Regression. [Link](https://www.cs.cmu.edu/~epxing/Class/10715/lectures/lecture2-LR.pdf)
+ L0, L1 and L2 regularization (Subset Selection, Lasso and Ridge Regression), a comparison. The Elements of
Statistical Learning, Trevor Hastie, Robert Tibshirani, Jerome Friedman, 2nd Edition, Section 3.4.3 [Link](https://web.stanford.edu/~hastie/ElemStatLearn/)
+ 
## Deep Learning
+ Why tanh for Recurrent Networks [Link](https://www.cs.cmu.edu/~bhiksha/courses/deeplearning/Fall.2015/slides/lec10.recurrent.pdf)
+ Receptive Fields in CNNs [Link](https://medium.com/mlreview/a-guide-to-receptive-field-arithmetic-for-convolutional-neural-networks-e0f514068807)
+ For everything Convolution [Link](https://arxiv.org/pdf/1603.07285.pdf) 
+ For eveything Gradient Descent [Link](https://ruder.io/optimizing-gradient-descent/)
+ Adaptive Learning rates in SGD [Link](https://www.cs.cornell.edu/courses/cs6787/2019fa/lectures/Lecture8.pdf)
+ Backpropagation in Python, Andrej Karpathy [Link](https://medium.com/@karpathy/yes-you-should-understand-backprop-e2f06eab496b)

## Probability and Statistics
+ As the title would say, "Generalized Linear Models, abridged".[Link](https://bwlewis.github.io/GLM/)
+ A good course to cover Statistics [Link](https://ocw.mit.edu/courses/mathematics/18-650-statistics-for-applications-fall-2016/lecture-slides/) 
+ Basic Statistics: Introduction to Mathematical Statistics, Hogg, McKean and Craig, Chapters 1-4. [Link](https://www.amazon.com/Introduction-Mathematical-Statistics-Robert-Hogg/dp/0321795431)
+ Introduction to Hypothesis Testing: Introduction to Mathematical Statistics, Hogg, McKean and Craig, Section 4.5-4.6 [Link](https://www.amazon.com/Introduction-Mathematical-Statistics-Robert-Hogg/dp/0321795431)
+ Examples of Uncorrelated vs Independent Random Variable [Link](https://www.stat.cmu.edu/~cshalizi/uADA/13/reminders/uncorrelated-vs-independent.pdf)
+ Discrete time Markov Chains,Poisson Processes, Renewal Theory Adventures in Stochastic Processes, 2nd Edition, Sidney Resnick [Link](http://do.unicyb.kiev.ua/iksan/lectures/Adventures.pdf) TODO: Add a link to more succint notes.
+ Q-Q Plots [Link](https://en.wikipedia.org/wiki/Q%E2%80%93Q_plot)

## Large Scale Machine Learning
+ Distributed version of several algorithms. https://10605.github.io/spring2020/

## Assorted Mathematics
+ Some facts about Symmetric Matrices. [Link](http://www.doc.ic.ac.uk/~ae/papers/lecture05.pdf)
+ Bare minimum SVD by Gilbert Strang. [Link](https://mitocw.ups.edu.ec/courses/mathematics/18-06sc-linear-algebra-fall-2011/positive-definite-matrices-and-applications/singular-value-decomposition/MIT18_06SCF11_Ses3.5sum.pdf)
+ SVD and PCA in real-life. [Link](https://jeremykun.com/2011/07/27/eigenfaces/)
+ If you are not sure how SVD and PCA are related. [Link](https://stats.stackexchange.com/questions/134282/relationship-between-svd-and-pca-how-to-use-svd-to-perform-pca)
+ If you want to brush up on Chain Rule (or if you are like me and get confused between gradient and derivative notation) [Link](http://www.met.reading.ac.uk/~ross/Documents/Chain.pdf).
 [Wikipedia](https://en.wikipedia.org/wiki/Gradient#Derivative) has some useful information as well.
+ Collection of Quantitative Interview problems by Pete Benson, University of Michigan. [Link](https://pbenson.github.io/docs/quantTechnicalQuestions/quantTechnicalQuestions.pdf)
+ Cholesky Factorization [Link](http://www.math.sjsu.edu/~foster/m143m/cholesky.pdf)
+ QR Factorization [Link](https://www.math.usm.edu/lambers/mat610/sum10/lecture9.pdf)

## System Design (for ML) 
+ Structure for Google Index [Link](http://infolab.stanford.edu/~backrub/google.html)
+ Recommender Systems, Xavier Amatriain [Link 1](https://www.youtube.com/watch?v=mRToFXlNBpQ)
[Link 2](https://www.youtube.com/watch?v=bLhq63ygoU8)
+ News Feed Ranking @ Facebook (Lars Backstrom) [Link](https://www.youtube.com/watch?v=Xpx5RYNTQvg)

## Uncategorized
+ Sobel Operator [Link](https://en.wikipedia.org/wiki/Sobel_operator)
+ You have a fair Die, and you can choose to roll it up to 3 times. Whenever you decide to stop, the number that’s facing up is your score. What strategy would you choose to maximize your score? What is the expected score with your strategy? If you are given more than 3 chances, can you improve your score?


# Cracking the Data Science Interview

![Cover Page](coverpage.jpg)

Welcome to the *Cracking the Data Science Interview* Github page. Here you will find data science related links, tutorials, blog posts, code snippets, interview prep material, case studies, and more! Have fun!


# Data Science Interview Preparation Materials

# General
- [How Do I Prepare For a Data Science Interview (Quora)](https://www.quora.com/How-do-I-prepare-for-a-data-scientist-interview)

- [How to Ace a Data Science Interview (Blog)](https://alyaabbott.wordpress.com/2014/10/01/how-to-ace-a-data-science-interview/)

- [How To Learn Data Science If You’re Broke (Towards Data Science)](https://towardsdatascience.com/how-to-learn-data-science-if-youre-broke-7ecc408b53c7)

- [Data Science Interview Questions (PDF)](https://rstudio-pubs-static.s3.amazonaws.com/172473_91262a8a4188445a8b5e81d5d31c7731.html)

- [120 Data Science Interview Questions](https://github.com/kojino/120-Data-Science-Interview-Questions)

- [Kaggle Kernels](https://www.kaggle.com/kernels)

- [Data Science Cheatsheets (Github Repo)](https://github.com/abhat222/Data-Science--Cheat-Sheet)



# Online Resources for Practice

- [Leetcode (Over 1350 Qustions To Practice Coding)](https://leetcode.com)

- [HackerRank (Coding)](https://www.hackerrank.com/home?utm_expid=.2u09ecQTSny1HV02SEVoCg.1&utm_referrer=https%3A%2F%2Fwww.google.com%2F)

- [SQLZoo (Place to Practice SQL)](https://sqlzoo.net)

- [SQLCourse- Interactive Online SQL Training](http://www.sqlcourse.com)


# Data Science Interview Prep Material

## Mathematical Prequisites
### Statistics
- [Statistics for Data Science (Blog)](https://blog.floydhub.com/statistics-for-data-science/)

- [The Math Behind A/B Testing with Example Python Code (Towards Data Science)](https://towardsdatascience.com/the-math-behind-a-b-testing-with-example-code-part-1-of-2-7be752e1d06f)


### Probability
- [Probability Cheatsheet (PDF)](http://www.wzchen.com/s/probability_cheatsheet.pdf), [(Github Repo)](https://github.com/wzchen/probability_cheatsheet)

- [Basics of Probability for Data Science explained with examples in R (Analytics Vidhya)](https://www.analyticsvidhya.com/blog/2017/02/basic-probability-data-science-with-examples/?source=post_page-----2db4f651bd63----------------------)

- [What is an intuitive explanation of Bayes' Rule? (Quora)](https://www.quora.com/What-is-an-intuitive-explanation-of-Bayes-Rule)


### Linear Algebra
- [Linear Algebra Cheat Sheet for Deep Learning (Towards Data Science)](https://towardsdatascience.com/linear-algebra-cheat-sheet-for-deep-learning-cd67aba4526c)

- [No Bullshit Guide to Linear ALgebra- Linear algebra explained in four pages (PDF)](https://www.souravsengupta.com/cds2016/lectures/Savov_Notes.pdf)

## Computer Science

### Data Structures
- [A Data Scientist’s Guide to Data Structures & Algorithms, Part 1 (Towards Data Science)](https://towardsdatascience.com/a-data-scientists-guide-to-data-structures-algorithms-1176395015a0)

- [A Data Scientist’s Guide to Data Structures & Algorithms, Part 2 (Towards Data Science)](https://towardsdatascience.com/a-data-scientists-guide-to-data-structures-algorithms-part-2-6bc27066f3fe)

- [Python Data Science Handbook](https://github.com/jakevdp/PythonDataScienceHandbook)

### Algorithms
- [Top Algorithms/Data Structures/Concepts every computer science student should know](https://medium.com/@codingfreak/top-algorithms-data-structures-concepts-every-computer-science-student-should-know-e0549c67b4ac)

### Databases
- [CAP Theorem (Wikipedia)](https://en.wikipedia.org/wiki/CAP_theorem)
- [Choosing The Right Database (Towards Data Science)](https://towardsdatascience.com/choosing-the-right-database-c45cd3a28f77)
### SQL
- [How To Ace Data Science Interviews: SQL (Towards Data Science)](https://towardsdatascience.com/how-to-ace-data-science-interviews-sql-b71de212e433)

### Python Packages/Libraries

- [Pandas](https://pandas.pydata.org/)

- [NumPy](http://www.numpy.org/)

- [SciPy](https://www.scipy.org/index.html)

- [Scikit-learn](https://scikit-learn.org/stable/)

- [Statsmodels](http://www.statsmodels.org/stable/index.html#)

- [PySpark](https://spark.apache.org/docs/latest/api/python/index.html)

- [Matplotlib](http://www.statsmodels.org/stable/index.html#)

- [IPython](http://ipython.org)

- [SymPy](https://www.sympy.org/en/index.html)

## Data Wrangling
- [Fundamental Techniques of Feature Engineering for Machine Learning](https://towardsdatascience.com/feature-engineering-for-machine-learning-3a5e293a5114)

- [The Ultimate Guide to Data Cleaning](https://towardsdatascience.com/the-ultimate-guide-to-data-cleaning-3969843991d4)







## Machine Learning

- [A Tour of Machine Learning Algorithms (Blog)](https://machinelearningmastery.com/a-tour-of-machine-learning-algorithms/)

- [41 Essential Machine Learning Interview Questions (Blog)](https://www.springboard.com/blog/machine-learning-interview-questions/?source=post_page-----2db4f651bd63----------------------)
### Supervised Learning Algorithms

- [Linear Regression — Detailed View](https://towardsdatascience.com/linear-regression-detailed-view-ea73175f6e86)

- [7 Regression Techniques you should know!](https://www.analyticsvidhya.com/blog/2015/08/comprehensive-guide-regression/)

- [Naive Bayes Classification — Theory](https://medium.com/machine-learning-101/chapter-1-supervised-learning-and-naive-bayes-classification-part-1-theory-8b9e361897d5)

- [SVM (Support Vector Machine) — Theory](https://medium.com/machine-learning-101/chapter-2-svm-support-vector-machine-theory-f0812effc72)

- [Decision Trees - Explained, Demystified and Simplified](https://adityashrm21.github.io/Decision-Trees/)

- [An Implementation and Explanation of the Random Forest in Python](https://towardsdatascience.com/an-implementation-and-explanation-of-the-random-forest-in-python-77bf308a9b76)

- [A Kaggle Master Explains Gradient Boosting](http://blog.kaggle.com/2017/01/23/a-kaggle-master-explains-gradient-boosting/)

- [A Gentle Introduction to XGBoost for Applied Machine Learning](https://machinelearningmastery.com/gentle-introduction-xgboost-applied-machine-learning/)

### Unsupervised Learning Algorithms

- [Unsupervised Learning and Data Clustering](https://towardsdatascience.com/unsupervised-learning-and-data-clustering-eeecb78b422a)

- [Understanding K-means Clustering in Machine Learning](https://towardsdatascience.com/understanding-k-means-clustering-in-machine-learning-6a6e67336aa1)

- [Hierarchical Clustering](https://www.saedsayad.com/clustering_hierarchical.htm)

- [Introduction to Autoencoders](https://www.jeremyjordan.me/autoencoders/)

- [An Intuitive Introduction to Generative Adversarial Networks](http://blog.kaggle.com/2018/01/18/an-intuitive-introduction-to-generative-adversarial-networks/)


### Reinforcement Learning Algorithms

- [Applications of Reinforcement Learning in Real World (Towards Data Science)](https://towardsdatascience.com/applications-of-reinforcement-learning-in-real-world-1a94955bcd12)

- [Open AI Gym](https://gym.openai.com)

- [Simple Reinforcement Learning Methods to Learn CartPole (Blog)](http://kvfrans.com/simple-algoritms-for-solving-cartpole/)

- [An introduction to Policy Gradients with Cartpole and Doom (freeCodeCamp)](https://www.freecodecamp.org/news/an-introduction-to-policy-gradients-with-cartpole-and-doom-495b5ef2207f/)

- [Implementation of Reinforcement Learning Algorithms (Github Repo)](https://github.com/dennybritz/reinforcement-learning)

# Miscellaneous

- [A Gentle Introduction To Graph Theory](https://medium.com/basecs/a-gentle-introduction-to-graph-theory-77969829ead8)

- [How to get started with machine learning on graphs](https://medium.com/octavian-ai/how-to-get-started-with-machine-learning-on-graphs-7f0795c83763)

- [Solving the Knapsack Problem with Dynamic Programming](https://dev.to/downey/solving-the-knapsack-problem-with-dynamic-programming-4hce)

- [An Overview of Monte Carlo Methods](https://towardsdatascience.com/an-overview-of-monte-carlo-methods-675384eb1694)

- [Introductory Guide on Linear Programming](https://www.analyticsvidhya.com/blog/2017/02/lintroductory-guide-on-linear-programming-explained-in-simple-english/)

- [ARIMA Models](http://www.forecastingsolutions.com/arima.html)

# Product 
- [Key Performance Indicators](https://www.shopify.com/blog/7365564-32-key-performance-indicators-kpis-for-ecommerce)


# Data Science Case Studies
- [Uber Data Science](https://www.uber.com/us/en/careers/teams/data-science/)

- [BCG Gamma](https://www.bcg.com/beyond-consulting/bcg-gamma/default.aspx)

- [McKinsey Solutions](https://www.mckinsey.com/solutions)

- [Bain Advanced Analytics](https://www.bain.com/vector-digital/advanced-analytics/)

- [Facebook Data Science](https://research.fb.com/teams/core-data-science/)

- [Kaggle](https://www.kaggle.com)

# Books
- [Cracking the Coding Interview](https://www.amazon.com/Cracking-Coding-Interview-Programming-Questions/dp/0984782850/ref=sr_1_1?keywords=cracking+the+co&qid=1575944295&sr=8-1)

- [The Hundred-Page Machine Learning Book ](https://www.amazon.com/Hundred-Page-Machine-Learning-Book/dp/199957950X/ref=pd_lutyp_crtyp_simh_1_7?_encoding=UTF8&pd_rd_i=199957950X&pd_rd_r=fbee59df-96ce-45b6-a41d-dcb9a769ad94&pd_rd_w=Dpj0r&pd_rd_wg=EokMX&pf_rd_p=11bf186d-590b-449a-8161-5414a5d28305&pf_rd_r=X99FPRSYC7DW3CXYJHCK&psc=1&refRID=X99FPRSYC7DW3CXYJHCK)

- [Deep Learning](http://www.deeplearningbook.org)

- [Practical Statistics for Data Scientists: 50 Essential Concepts](https://www.amazon.com/Practical-Statistics-Data-Scientists-Essential-ebook/dp/B071NVDFD6/ref=pd_sim_351_1/147-3762810-2360818?_encoding=UTF8&pd_rd_i=B071NVDFD6&pd_rd_r=52496a3c-32d4-4d5c-a873-0b832ff9b0a4&pd_rd_w=FohrU&pd_rd_wg=lIxdZ&pf_rd_p=04d27813-a1f2-4e7b-a32b-b5ab374ce3f9&pf_rd_r=NWTQ82150B8GMD6P43DD&psc=1&refRID=NWTQ82150B8GMD6P43DD)

- [Introduction to Algorithms](https://www.amazon.com/Introduction-Algorithms-3rd-MIT-Press/dp/0262033844)

# Blogs

- [Machine Learning Mastery](https://machinelearningmastery.com)

- [GeeksforGeeks](https://www.geeksforgeeks.org)

- [No Free Hunch- The Official Blog of Kaggle.com](http://blog.kaggle.com)

- [Towards Data Science](https://towardsdatascience.com)

- [The Art of Data Science](https://www.quora.com/q/art-of-data-science)


<a href="https://www.buymeacoffee.com/khalel" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" style="height: 60px !important;width: 217px !important;" ></a>

# Machine Learning Interview Questions
A collection of technical interview questions for machine learning and computer vision engineering positions.

#### 1) What's the trade-off between bias and variance? [[src](http://houseofbots.com/news-detail/2849-4-data-science-and-machine-learning-interview-questions)]

If our model is too simple and has very few parameters then it may have high bias and low variance. On the other hand if our model has large number of parameters then it’s going to have high variance and low bias. So we need to find the right/good balance without overfitting and underfitting the data. [[src]](https://towardsdatascience.com/understanding-the-bias-variance-tradeoff-165e6942b229)

#### 2) What is gradient descent? [[src](http://houseofbots.com/news-detail/2849-4-data-science-and-machine-learning-interview-questions)]
[[Answer]](https://machinelearningmastery.com/gradient-descent-for-machine-learning/)

Gradient descent is an optimization algorithm used to find the values of parameters (coefficients) of a function (f) that minimizes a cost function (cost).

Gradient descent is best used when the parameters cannot be calculated analytically (e.g. using linear algebra) and must be searched for by an optimization algorithm.

#### 3) Explain over- and under-fitting and how to combat them? [[src](http://houseofbots.com/news-detail/2849-4-data-science-and-machine-learning-interview-questions)]
[[Answer]](https://towardsdatascience.com/overfitting-vs-underfitting-a-complete-example-d05dd7e19765)

#### 4) How do you combat the curse of dimensionality? [[src](http://houseofbots.com/news-detail/2849-4-data-science-and-machine-learning-interview-questions)]

 - Manual Feature Selection
 - Principal Component Analysis (PCA)
 - Multidimensional Scaling
 - Locally linear embedding  
[[src]](https://towardsdatascience.com/why-and-how-to-get-rid-of-the-curse-of-dimensionality-right-with-breast-cancer-dataset-7d528fb5f6c0)

#### 5) What is regularization, why do we use it, and give some examples of common methods? [[src](http://houseofbots.com/news-detail/2849-4-data-science-and-machine-learning-interview-questions)]
A technique that discourages learning a more complex or flexible model, so as to avoid the risk of overfitting. 
Examples
 - Ridge (L2 norm)
 - Lasso (L1 norm)  
The obvious *disadvantage* of **ridge** regression, is model interpretability. It will shrink the coefficients for least important predictors, very close to zero. But it will never make them exactly zero. In other words, the *final model will include all predictors*. However, in the case of the **lasso**, the L1 penalty has the effect of forcing some of the coefficient estimates to be *exactly equal* to zero when the tuning parameter λ is sufficiently large. Therefore, the lasso method also performs variable selection and is said to yield sparse models.
[[src]](https://towardsdatascience.com/regularization-in-machine-learning-76441ddcf99a)

#### 6) Explain Principal Component Analysis (PCA)? [[src](http://houseofbots.com/news-detail/2849-4-data-science-and-machine-learning-interview-questions)]
[[Answer]](https://towardsdatascience.com/a-one-stop-shop-for-principal-component-analysis-5582fb7e0a9c)

#### 7) Why is ReLU better and more often used than Sigmoid in Neural Networks? [[src](http://houseofbots.com/news-detail/2849-4-data-science-and-machine-learning-interview-questions)]
Imagine a network with random initialized weights ( or normalised ) and almost 50% of the network yields 0 activation because of the characteristic of ReLu ( output 0 for negative values of x ). This means a fewer neurons are firing ( sparse activation ) and the network is lighter. [[src]](https://medium.com/the-theory-of-everything/understanding-activation-functions-in-neural-networks-9491262884e0)


#### 8) Given stride S and kernel sizes  for each layer of a (1-dimensional) CNN, create a function to compute the [receptive field](https://www.quora.com/What-is-a-receptive-field-in-a-convolutional-neural-network) of a particular node in the network. This is just finding how many input nodes actually connect through to a neuron in a CNN. [[src](https://www.reddit.com/r/computervision/comments/7gku4z/technical_interview_questions_in_cv/)]

The receptive field are defined portion of space within an inputs that will be used during an operation to generate an output.

Considering a CNN filter of size k, the receptive field of a peculiar layer is only the number of input used by the filter, in this case k, multiplied by the dimension of the input that is not being reduced by the convolutionnal filter a. This results in a receptive field of k*a.

More visually, in the case of an image of size 32x32x3, with a CNN with a filter size of 5x5, the corresponding recpetive field will be the the filter size, 5 multiplied by the depth of the input volume (the RGB colors) which is the color dimensio. This thus gives us a recpetive field of dimension 5x5x3.

#### 9) Implement [connected components](http://aishack.in/tutorials/labelling-connected-components-example/) on an image/matrix. [[src](https://www.reddit.com/r/computervision/comments/7gku4z/technical_interview_questions_in_cv/)]


#### 10) Implement a sparse matrix class in C++. [[src](https://www.reddit.com/r/computervision/comments/7gku4z/technical_interview_questions_in_cv/)]

[[Answer]](https://www.geeksforgeeks.org/sparse-matrix-representation/)

#### 11) Create a function to compute an [integral image](https://en.wikipedia.org/wiki/Summed-area_table), and create another function to get area sums from the integral image.[[src](https://www.reddit.com/r/computervision/comments/7gku4z/technical_interview_questions_in_cv/)]

[[Answer]](https://www.geeksforgeeks.org/submatrix-sum-queries/)

#### 12) How would you remove outliers when trying to estimate a flat plane from noisy samples? [[src](https://www.reddit.com/r/computervision/comments/7gku4z/technical_interview_questions_in_cv/)]

Random sample consensus (RANSAC) is an iterative method to estimate parameters of a mathematical model from a set of observed data that contains outliers, when outliers are to be accorded no influence on the values of the estimates.
[[src]](https://en.wikipedia.org/wiki/Random_sample_consensus)



#### 13) How does [CBIR](https://www.robots.ox.ac.uk/~vgg/publications/2013/arandjelovic13/arandjelovic13.pdf) work? [[src](https://www.reddit.com/r/computervision/comments/7gku4z/technical_interview_questions_in_cv/)]

[[Answer]](https://en.wikipedia.org/wiki/Content-based_image_retrieval)
Content-based image retrieval is the concept of using images to gather metadata on their content. Compared to the current image retrieval approach based on the keywords associated to the images, this technique generates its metadata from computer vision techniques to extract the relevant informations that will be used during the querying step. Many approach are possible from feature detection to retrieve keywords to the usage of CNN to extract dense features that will be associated to a known distribution of keywords. 

With this last approach, we care less about what is shown on the image but more about the similarity between the metadata generated by a known image and a list of known label and or tags projected into this metadata space.

#### 14) How does image registration work? Sparse vs. dense [optical flow](http://www.ncorr.com/download/publications/bakerunify.pdf) and so on. [[src](https://www.reddit.com/r/computervision/comments/7gku4z/technical_interview_questions_in_cv/)]

#### 15) Describe how convolution works. What about if your inputs are grayscale vs RGB imagery? What determines the shape of the next layer? [[src](https://www.reddit.com/r/computervision/comments/7gku4z/technical_interview_questions_in_cv/)]

[[Answer]](https://dev.to/sandeepbalachandran/machine-learning-convolution-with-color-images-2p41)

#### 16) Talk me through how you would create a 3D model of an object from imagery and depth sensor measurements taken at all angles around the object. [[src](https://www.reddit.com/r/computervision/comments/7gku4z/technical_interview_questions_in_cv/)]

#### 17) Implement SQRT(const double & x) without using any special functions, just fundamental arithmetic. [[src](https://www.reddit.com/r/computervision/comments/7gku4z/technical_interview_questions_in_cv/)]

The taylor series can be used for this step by providing an approximation of sqrt(x):

[[Answer]](https://math.stackexchange.com/questions/732540/taylor-series-of-sqrt1x-using-sigma-notation)

#### 18) Reverse a bitstring. [[src](https://www.reddit.com/r/computervision/comments/7gku4z/technical_interview_questions_in_cv/)]

If you are using python3 :

```
data = b'\xAD\xDE\xDE\xC0'
my_data = bytearray(data)
my_data.reverse()
```
#### 19) Implement non maximal suppression as efficiently as you can. [[src](https://www.reddit.com/r/computervision/comments/7gku4z/technical_interview_questions_in_cv/)]



#### 20) Reverse a linked list in place. [[src](https://www.reddit.com/r/computervision/comments/7gku4z/technical_interview_questions_in_cv/)]

[[Answer]](https://www.geeksforgeeks.org/reverse-a-linked-list/)

#### 21) What is data normalization and why do we need it? [[src](http://houseofbots.com/news-detail/2849-4-data-science-and-machine-learning-interview-questions)]
Data normalization is very important preprocessing step, used to rescale values to fit in a specific range to assure better convergence during backpropagation. In general, it boils down to subtracting the mean of each data point and dividing by its standard deviation. If we don't do this then some of the features (those with high magnitude) will be weighted more in the cost function (if a higher-magnitude feature changes by 1%, then that change is pretty big, but for smaller features it's quite insignificant). The data normalization makes all features weighted equally.

#### 22) Why do we use convolutions for images rather than just FC layers? [[src](http://houseofbots.com/news-detail/2849-4-data-science-and-machine-learning-interview-questions)]
Firstly, convolutions preserve, encode, and actually use the spatial information from the image. If we used only FC layers we would have no relative spatial information. Secondly, Convolutional Neural Networks (CNNs) have a partially built-in translation in-variance, since each convolution kernel acts as it's own filter/feature detector.

#### 23) What makes CNNs translation invariant? [[src](http://houseofbots.com/news-detail/2849-4-data-science-and-machine-learning-interview-questions)]
As explained above, each convolution kernel acts as it's own filter/feature detector. So let's say you're doing object detection, it doesn't matter where in the image the object is since we're going to apply the convolution in a sliding window fashion across the entire image anyways.

#### 24) Why do we have max-pooling in classification CNNs? [[src](http://houseofbots.com/news-detail/2849-4-data-science-and-machine-learning-interview-questions)]
for a role in Computer Vision. Max-pooling in a CNN allows you to reduce computation since your feature maps are smaller after the pooling. You don't lose too much semantic information since you're taking the maximum activation. There's also a theory that max-pooling contributes a bit to giving CNNs more translation in-variance. Check out this great video from Andrew Ng on the [benefits of max-pooling](https://www.coursera.org/learn/convolutional-neural-networks/lecture/hELHk/pooling-layers).

#### 25) Why do segmentation CNNs typically have an encoder-decoder style / structure? [[src](http://houseofbots.com/news-detail/2849-4-data-science-and-machine-learning-interview-questions)]
The encoder CNN can basically be thought of as a feature extraction network, while the decoder uses that information to predict the image segments by "decoding" the features and upscaling to the original image size.

#### 26) What is the significance of Residual Networks? [[src](http://houseofbots.com/news-detail/2849-4-data-science-and-machine-learning-interview-questions)]
The main thing that residual connections did was allow for direct feature access from previous layers. This makes information propagation throughout the network much easier. One very interesting paper about this shows how using local skip connections gives the network a type of ensemble multi-path structure, giving features multiple paths to propagate throughout the network.

#### 27) What is batch normalization and why does it work? [[src](http://houseofbots.com/news-detail/2849-4-data-science-and-machine-learning-interview-questions)]
Training Deep Neural Networks is complicated by the fact that the distribution of each layer's inputs changes during training, as the parameters of the previous layers change. The idea is then to normalize the inputs of each layer in such a way that they have a mean output activation of zero and standard deviation of one. This is done for each individual mini-batch at each layer i.e compute the mean and variance of that mini-batch alone, then normalize. This is analogous to how the inputs to networks are standardized. How does this help? We know that normalizing the inputs to a network helps it learn. But a network is just a series of layers, where the output of one layer becomes the input to the next. That means we can think of any layer in a neural network as the first layer of a smaller subsequent network. Thought of as a series of neural networks feeding into each other, we normalize the output of one layer before applying the activation function, and then feed it into the following layer (sub-network).

#### 28) Why would you use many small convolutional kernels such as 3x3 rather than a few large ones? [[src](http://houseofbots.com/news-detail/2849-4-data-science-and-machine-learning-interview-questions)]
This is very well explained in the [VGGNet paper](https://arxiv.org/pdf/1409.1556.pdf). There are 2 reasons: First, you can use several smaller kernels rather than few large ones to get the same receptive field and capture more spatial context, but with the smaller kernels you are using less parameters and computations. Secondly, because with smaller kernels you will be using more filters, you'll be able to use more activation functions and thus have a more discriminative mapping function being learned by your CNN.

#### 29) Why do we need a validation set and test set? What is the difference between them? [[src](https://www.toptal.com/machine-learning/interview-questions)]
When training a model, we divide the available data into three separate sets:

 - The training dataset is used for fitting the model’s parameters. However, the accuracy that we achieve on the training set is not reliable for predicting if the model will be accurate on new samples.
 - The validation dataset is used to measure how well the model does on examples that weren’t part of the training dataset. The metrics computed on the validation data can be used to tune the hyperparameters of the model. However, every time we evaluate the validation data and we make decisions based on those scores, we are leaking information from the validation data into our model. The more evaluations, the more information is leaked. So we can end up overfitting to the validation data, and once again the validation score won’t be reliable for predicting the behaviour of the model in the real world.
 - The test dataset is used to measure how well the model does on previously unseen examples. It should only be used once we have tuned the parameters using the validation set.

So if we omit the test set and only use a validation set, the validation score won’t be a good estimate of the generalization of the model.

#### 30) What is stratified cross-validation and when should we use it? [[src](https://www.toptal.com/machine-learning/interview-questions)]
Cross-validation is a technique for dividing data between training and validation sets. On typical cross-validation this split is done randomly. But in stratified cross-validation, the split preserves the ratio of the categories on both the training and validation datasets.

For example, if we have a dataset with 10% of category A and 90% of category B, and we use stratified cross-validation, we will have the same proportions in training and validation. In contrast, if we use simple cross-validation, in the worst case we may find that there are no samples of category A in the validation set.

Stratified cross-validation may be applied in the following scenarios:

 - On a dataset with multiple categories. The smaller the dataset and the more imbalanced the categories, the more important it will be to use stratified cross-validation.
 - On a dataset with data of different distributions. For example, in a dataset for autonomous driving, we may have images taken during the day and at night. If we do not ensure that both types are present in training and validation, we will have generalization problems.

#### 31) Why do ensembles typically have higher scores than individual models? [[src](https://www.toptal.com/machine-learning/interview-questions)]
An ensemble is the combination of multiple models to create a single prediction. The key idea for making better predictions is that the models should make different errors. That way the errors of one model will be compensated by the right guesses of the other models and thus the score of the ensemble will be higher.

We need diverse models for creating an ensemble. Diversity can be achieved by:
 - Using different ML algorithms. For example, you can combine logistic regression, k-nearest neighbors, and decision trees.
 - Using different subsets of the data for training. This is called bagging.
 - Giving a different weight to each of the samples of the training set. If this is done iteratively, weighting the samples according to the errors of the ensemble, it’s called boosting.
Many winning solutions to data science competitions are ensembles. However, in real-life machine learning projects, engineers need to find a balance between execution time and accuracy.

#### 32) What is an imbalanced dataset? Can you list some ways to deal with it? [[src](https://www.toptal.com/machine-learning/interview-questions)]
An imbalanced dataset is one that has different proportions of target categories. For example, a dataset with medical images where we have to detect some illness will typically have many more negative samples than positive samples—say, 98% of images are without the illness and 2% of images are with the illness.

There are different options to deal with imbalanced datasets:
 - Oversampling or undersampling. Instead of sampling with a uniform distribution from the training dataset, we can use other distributions so the model sees a more balanced dataset.
 - Data augmentation. We can add data in the less frequent categories by modifying existing data in a controlled way. In the example dataset, we could flip the images with illnesses, or add noise to copies of the images in such a way that the illness remains visible.
 - Using appropriate metrics. In the example dataset, if we had a model that always made negative predictions, it would achieve a precision of 98%. There are other metrics such as precision, recall, and F-score that describe the accuracy of the model better when using an imbalanced dataset.

#### 33) Can you explain the differences between supervised, unsupervised, and reinforcement learning? [[src](https://www.toptal.com/machine-learning/interview-questions)]
In supervised learning, we train a model to learn the relationship between input data and output data. We need to have labeled data to be able to do supervised learning.

With unsupervised learning, we only have unlabeled data. The model learns a representation of the data. Unsupervised learning is frequently used to initialize the parameters of the model when we have a lot of unlabeled data and a small fraction of labeled data. We first train an unsupervised model and, after that, we use the weights of the model to train a supervised model.

In reinforcement learning, the model has some input data and a reward depending on the output of the model. The model learns a policy that maximizes the reward. Reinforcement learning has been applied successfully to strategic games such as Go and even classic Atari video games.

#### 34) What is data augmentation? Can you give some examples? [[src](https://www.toptal.com/machine-learning/interview-questions)]
Data augmentation is a technique for synthesizing new data by modifying existing data in such a way that the target is not changed, or it is changed in a known way.

Computer vision is one of fields where data augmentation is very useful. There are many modifications that we can do to images:
 - Resize
 - Horizontal or vertical flip
 - Rotate
 - Add noise
 - Deform
 - Modify colors
Each problem needs a customized data augmentation pipeline. For example, on OCR, doing flips will change the text and won’t be beneficial; however, resizes and small rotations may help.

#### 35) What is Turing test? [[src](https://intellipaat.com/interview-question/artificial-intelligence-interview-questions/)]
The Turing test is a method to test the machine’s ability to match the human level intelligence. A machine is used to challenge the human intelligence that when it passes the test, it is considered as intelligent. Yet a machine could be viewed as intelligent without sufficiently knowing about people to mimic a human.

#### 36) What is Precision?  
Precision (also called positive predictive value) is the fraction of relevant instances among the retrieved instances  
Precision = true positive / (true positive + false positive)  
[[src]](https://en.wikipedia.org/wiki/Precision_and_recall)

#### 37) What is Recall?  
Recall (also known as sensitivity) is the fraction of relevant instances that have been retrieved over the total amount of relevant instances.
Recall = true positive / (true positive + false negative)  
[[src]](https://en.wikipedia.org/wiki/Precision_and_recall)

#### 38) Define F1-score. [[src](https://intellipaat.com/interview-question/artificial-intelligence-interview-questions/)]
It is the weighted average of precision and recall. It considers both false positive and false negative into account. It is used to measure the model’s performance.  
F1-Score = 2 * (precision * recall) / (precision + recall)

#### 39) What is cost function? [[src](https://intellipaat.com/interview-question/artificial-intelligence-interview-questions/)]
Cost function is a scalar functions which Quantifies the error factor of the Neural Network. Lower the cost function better the Neural network. Eg: MNIST Data set to classify the image, input image is digit 2 and the Neural network wrongly predicts it to be 3

#### 40) List different activation neurons or functions. [[src](https://intellipaat.com/interview-question/artificial-intelligence-interview-questions/)]
 - Linear Neuron
 - Binary Threshold Neuron
 - Stochastic Binary Neuron
 - Sigmoid Neuron
 - Tanh function
 - Rectified Linear Unit (ReLU)

#### 41) Define Learning Rate.
Learning rate is a hyper-parameter that controls how much we are adjusting the weights of our network with respect the loss gradient. [[src](https://en.wikipedia.org/wiki/Learning_rate)]

#### 42) What is Momentum (w.r.t NN optimization)?
Momentum lets the optimization algorithm remembers its last step, and adds some proportion of it to the current step. This way, even if the algorithm is stuck in a flat region, or a small local minimum, it can get out and continue towards the true minimum. [[src]](https://www.quora.com/What-is-the-difference-between-momentum-and-learning-rate)

#### 43) What is the difference between Batch Gradient Descent and Stochastic Gradient Descent?
Batch gradient descent computes the gradient using the whole dataset. This is great for convex, or relatively smooth error manifolds. In this case, we move somewhat directly towards an optimum solution, either local or global. Additionally, batch gradient descent, given an annealed learning rate, will eventually find the minimum located in it's basin of attraction.

Stochastic gradient descent (SGD) computes the gradient using a single sample. SGD works well (Not well, I suppose, but better than batch gradient descent) for error manifolds that have lots of local maxima/minima. In this case, the somewhat noisier gradient calculated using the reduced number of samples tends to jerk the model out of local minima into a region that hopefully is more optimal. [[src]](https://stats.stackexchange.com/questions/49528/batch-gradient-descent-versus-stochastic-gradient-descent)

#### 44) Epoch vs. Batch vs. Iteration.
 - **Epoch**: one forward pass and one backward pass of **all** the training examples  
 - **Batch**: examples processed together in one pass (forward and backward)  
 - **Iteration**: number of training examples / Batch size  

#### 45) What is vanishing gradient? [[src](https://intellipaat.com/interview-question/artificial-intelligence-interview-questions/)]
As we add more and more hidden layers, back propagation becomes less and less useful in passing information to the lower layers. In effect, as information is passed back, the gradients begin to vanish and become small relative to the weights of the networks.

#### 46) What are dropouts? [[src](https://intellipaat.com/interview-question/artificial-intelligence-interview-questions/)]
Dropout is a simple way to prevent a neural network from overfitting. It is the dropping out of some of the units in a neural network. It is similar to the natural reproduction process, where the nature produces offsprings by combining distinct genes (dropping out others) rather than strengthening the co-adapting of them.

#### 47) Define LSTM. [[src](https://intellipaat.com/interview-question/artificial-intelligence-interview-questions/)]
Long Short Term Memory – are explicitly designed to address the long term dependency problem, by maintaining a state what to remember and what to forget.

#### 48) List the key components of LSTM. [[src](https://intellipaat.com/interview-question/artificial-intelligence-interview-questions/)]
 - Gates (forget, Memory, update & Read)
 - tanh(x) (values between -1 to 1)
 - Sigmoid(x) (values between 0 to 1)

#### 49) List the variants of RNN. [[src](https://intellipaat.com/interview-question/artificial-intelligence-interview-questions/)]
 - LSTM: Long Short Term Memory
 - GRU: Gated Recurrent Unit
 - End to End Network
 - Memory Network

#### 50) What is Autoencoder, name few applications. [[src](https://intellipaat.com/interview-question/artificial-intelligence-interview-questions/)]
Auto encoder is basically used to learn a compressed form of given data. Few applications include
 - Data denoising
 - Dimensionality reduction
 - Image reconstruction
 - Image colorization

#### 51) What are the components of GAN? [[src](https://intellipaat.com/interview-question/artificial-intelligence-interview-questions/)]
 - Generator
 - Discriminator

#### 52) What's the difference between boosting and bagging?
Boosting and bagging are similar, in that they are both ensembling techniques, where a number of weak learners (classifiers/regressors that are barely better than guessing) combine (through averaging or max vote) to create a strong learner that can make accurate predictions. Bagging means that you take bootstrap samples (with replacement) of your data set and each sample trains a (potentially) weak learner. Boosting, on the other hand, uses all data to train each learner, but instances that were misclassified by the previous learners are given more weight so that subsequent learners give more focus to them during training. [[src]](https://www.quora.com/Whats-the-difference-between-boosting-and-bagging)

#### 53) Explain how a ROC curve works. [[src]](https://www.springboard.com/blog/machine-learning-interview-questions/)
The ROC curve is a graphical representation of the contrast between true positive rates and the false positive rate at various thresholds. It’s often used as a proxy for the trade-off between the sensitivity of the model (true positives) vs the fall-out or the probability it will trigger a false alarm (false positives).

#### 54) What’s the difference between Type I and Type II error? [[src]](https://www.springboard.com/blog/machine-learning-interview-questions/)
Type I error is a false positive, while Type II error is a false negative. Briefly stated, Type I error means claiming something has happened when it hasn’t, while Type II error means that you claim nothing is happening when in fact something is.
A clever way to think about this is to think of Type I error as telling a man he is pregnant, while Type II error means you tell a pregnant woman she isn’t carrying a baby.

#### 55) What’s the difference between a generative and discriminative model? [[src]](https://www.springboard.com/blog/machine-learning-interview-questions/)
A generative model will learn categories of data while a discriminative model will simply learn the distinction between different categories of data. Discriminative models will generally outperform generative models on classification tasks.

#### 56) Instance-Based Versus Model-Based Learning.

 - **Instance-based Learning**: The system learns the examples by heart, then generalizes to new cases using a similarity measure.

 - **Model-based Learning**: Another way to generalize from a set of examples is to build a model of these examples, then use that model to make predictions. This is called model-based learning.
[[src]](https://medium.com/@sanidhyaagrawal08/what-is-instance-based-and-model-based-learning-s1e10-8e68364ae084)


#### 57) When to use a Label Encoding vs. One Hot Encoding?

This question generally depends on your dataset and the model which you wish to apply. But still, a few points to note before choosing the right encoding technique for your model:

We apply One-Hot Encoding when:

- The categorical feature is not ordinal (like the countries above)
- The number of categorical features is less so one-hot encoding can be effectively applied
We apply Label Encoding when:

- The categorical feature is ordinal (like Jr. kg, Sr. kg, Primary school, high school)
- The number of categories is quite large as one-hot encoding can lead to high memory consumption

[[src]](https://www.analyticsvidhya.com/blog/2020/03/one-hot-encoding-vs-label-encoding-using-scikit-learn/)

#### 58) What is the difference between LDA and PCA for dimensionality reduction?

Both LDA and PCA are linear transformation techniques: LDA is a supervised whereas PCA is unsupervised – PCA ignores class labels.

We can picture PCA as a technique that finds the directions of maximal variance. In contrast to PCA, LDA attempts to find a feature subspace that maximizes class separability.

[[src]](https://sebastianraschka.com/faq/docs/lda-vs-pca.html)

#### 59) What is t-SNE?

t-Distributed Stochastic Neighbor Embedding (t-SNE) is an unsupervised, non-linear technique primarily used for data exploration and visualizing high-dimensional data. In simpler terms, t-SNE gives you a feel or intuition of how the data is arranged in a high-dimensional space. 

[[src]](https://towardsdatascience.com/an-introduction-to-t-sne-with-python-example-5a3a293108d1)

#### 60) What is the difference between t-SNE and PCA for dimensionality reduction?

The first thing to note is that PCA was developed in 1933 while t-SNE was developed in 2008. A lot has changed in the world of data science since 1933 mainly in the realm of compute and size of data. Second, PCA is a linear dimension reduction technique that seeks to maximize variance and preserves large pairwise distances. In other words, things that are different end up far apart. This can lead to poor visualization especially when dealing with non-linear manifold structures. Think of a manifold structure as any geometric shape like: cylinder, ball, curve, etc.

t-SNE differs from PCA by preserving only small pairwise distances or local similarities whereas PCA is concerned with preserving large pairwise distances to maximize variance.

[[src]](https://towardsdatascience.com/an-introduction-to-t-sne-with-python-example-5a3a293108d1)

#### 61) What is UMAP?

UMAP (Uniform Manifold Approximation and Projection) is a novel manifold learning technique for dimension reduction. UMAP is constructed from a theoretical framework based in Riemannian geometry and algebraic topology. The result is a practical scalable algorithm that applies to real world data.

[[src]](https://arxiv.org/abs/1802.03426#:~:text=UMAP%20)

#### 62) What is the difference between t-SNE and UMAP for dimensionality reduction?

The biggest difference between the the output of UMAP when compared with t-SNE is this balance between local and global structure - UMAP is often better at preserving global structure in the final projection. This means that the inter-cluster relations are potentially more meaningful than in t-SNE. However, it's important to note that, because UMAP and t-SNE both necessarily warp the high-dimensional shape of the data when projecting to lower dimensions, any given axis or distance in lower dimensions still isn’t directly interpretable in the way of techniques such as PCA.

[[src]](https://pair-code.github.io/understanding-umap/)

#### 63) How Random Number Generator Works, e.g. rand() function in python works?
It generates a pseudo random number based on the seed and there are some famous algorithm, please see below link for further information on this.
[[src]](https://en.wikipedia.org/wiki/Linear_congruential_generator)


## Contributions
Contributions are most welcomed.
 1. Fork the repository.
 2. Commit your *questions* or *answers*.
 3. Open **pull request**.


[Source](https://docs.google.com/document/d/1ajyJhXyt4q9ZsufXV1kZxDH_3Isg3MYAKsFqNytXrCw/)

- [1. Why do you use feature selection?](#1-why-do-you-use-feature-selection)
    - [Filter Methods](#filter-methods)
    - [Embedded Methods](#embedded-methods)
    - [Misleading](#misleading)
    - [Overfitting](#overfitting)
- [2. Explain what regularization is and why it is useful.](#2-explain-what-regularization-is-and-why-it-is-useful)
- [3. What’s the difference between L1 and L2 regularization?](#3-whats-the-difference-between-l1-and-l2-regularization)
- [4. How would you validate a model you created to generate a predictive model of a quantitative outcome variable using multiple regression?](#4-how-would-you-validate-a-model-you-created-to-generate-a-predictive-model-of-a-quantitative-outcome-variable-using-multiple-regression)
- [5. Explain what precision and recall are. How do they relate to the ROC curve?](#5-explain-what-precision-and-recall-are-how-do-they-relate-to-the-roc-curve)
- [6. Is it better to have too many false positives, or too many false negatives?](#6-is-it-better-to-have-too-many-false-positives--or-too-many-false-negatives)
- [7. How do you deal with unbalanced binary classification?](#7-how-do-you-deal-with-unbalanced-binary-classification)
- [8. What is statistical power?](#8-what-is-statistical-power)
- [9. What are bias and variance, and what are their relation to modeling data?](#9-what-are-bias-and-variance--and-what-are-their-relation-to-modeling-data)
    - [Approaches](#approaches)
- [10. What if the classes are imbalanced? What if there are more than 2 groups?](#10-what-if-the-classes-are-imbalanced-what-if-there-are-more-than-2-groups)
- [11. What are some ways I can make my model more robust to outliers?](#11-what-are-some-ways-i-can-make-my-model-more-robust-to-outliers)
- [12. In unsupervised learning, if a ground truth about a dataset is unknown, how can we determine the most useful number of clusters to be?](#12-in-unsupervised-learning--if-a-ground-truth-about-a-dataset-is-unknown--how-can-we-determine-the-most-useful-number-of-clusters-to-be)
- [13. Define variance](#13-define-variance)
- [14. Expected value](#14-expected-value)
- [15. Describe the differences between and use cases for box plots and histograms](#15-describe-the-differences-between-and-use-cases-for-box-plots-and-histograms)
- [16. How would you find an anomaly in a distribution?](#16-how-would-you-find-an-anomaly-in-a-distribution)
    - [Statistical methods](#statistical-methods)
    - [Metric methods](#metric-methods)
- [17. How do you deal with outliers in your data?](#17-how-do-you-deal-with-outliers-in-your-data)
- [18. How do you deal with sparse data?](#18-how-do-you-deal-with-sparse-data)
- [19. Big Data Engineer Can you explain what REST is?](#19-big-data-engineer-can-you-explain-what-rest-is)
- [20. Logistic regression](#20-logistic-regression)
- [21. What is the effect on the coefficients of logistic regression if two predictors are highly correlated? What are the confidence intervals of the coefficients?](#21-what-is-the-effect-on-the-coefficients-of-logistic-regression-if-two-predictors-are-highly-correlated-what-are-the-confidence-intervals-of-the-coefficients)
- [22. What’s the difference between Gaussian Mixture Model and K-Means?](#22-whats-the-difference-between-gaussian-mixture-model-and-k-means)
- [23. Describe how Gradient Boosting works.](#23-describe-how-gradient-boosting-works)
  - [AdaBoost the First Boosting Algorithm](#adaboost-the-first-boosting-algorithm)
    - [Loss Function](#loss-function)
    - [Weak Learner](#weak-learner)
    - [Additive Model](#additive-model)
  - [Improvements to Basic Gradient Boosting](#improvements-to-basic-gradient-boosting)
    - [Tree Constraints](#tree-constraints)
    - [Weighted Updates](#weighted-updates)
    - [Stochastic Gradient Boosting](#stochastic-gradient-boosting)
    - [Penalized Gradient Boosting](#penalized-gradient-boosting)
- [24. Difference between AdaBoost and XGBoost](#24-difference-between-AdaBoost-and-XGBoost)
- [25. Data Mining Describe the decision tree model.](#25-data-mining-describe-the-decision-tree-model)
- [26. Notes from Coursera Deep Learning courses by Andrew Ng](#26-notes-from-coursera-deep-learning-courses-by-andrew-ng)
- [27. What is a neural network?](#27-what-is-a-neural-network)
- [28. How do you deal with sparse data?](#28-how-do-you-deal-with-sparse-data)
- [29. RNN and LSTM](#29-rnn-and-lstm)
- [30. Pseudo Labeling](#30-pseudo-labeling)
- [31. Knowledge Distillation](#31-knowledge-distillation)
- [32. What is an inductive bias?](#32-what-is-an-inductive-bias)
- [33. What is a confidence interval in layman's terms?](#33-confidence-interval-in-layman's-terms)


## 1. Why do you use feature selection?
Feature selection is the process of selecting a subset of relevant features for use in model construction. Feature selection is itself useful, but it mostly acts as a filter, muting out features that aren’t useful in addition to your existing features.
Feature selection methods aid you in your mission to create an accurate predictive model. They help you by choosing features that will give you as good or better accuracy whilst requiring less data.
Feature selection methods can be used to identify and remove unneeded, irrelevant and redundant attributes from data that do not contribute to the accuracy of a predictive model or may in fact decrease the accuracy of the model.
Fewer attributes is desirable because it reduces the complexity of the model, and a simpler model is simpler to understand and explain.
#### Filter Methods
Filter feature selection methods apply a statistical measure to assign a scoring to each feature. The features are ranked by the score and either selected to be kept or removed from the dataset. The methods are often univariate and consider the feature independently, or with regard to the dependent variable.
Some examples of some filter methods include the Chi squared test, information gain and correlation coefficient scores.
#### Embedded Methods
Embedded methods learn which features best contribute to the accuracy of the model while the model is being created. The most common type of embedded feature selection methods are regularization methods.
Regularization methods are also called penalization methods that introduce additional constraints into the optimization of a predictive algorithm (such as a regression algorithm) that bias the model toward lower complexity (fewer coefficients).
Examples of regularization algorithms are the LASSO, Elastic Net and Ridge Regression.
#### Misleading
Including redundant attributes can be misleading to modeling algorithms. Instance-based methods such as k-nearest neighbor use small neighborhoods in the attribute space to determine classification and regression predictions. These predictions can be greatly skewed by redundant attributes.
#### Overfitting
Keeping irrelevant attributes in your dataset can result in overfitting. Decision tree algorithms like C4.5 seek to make optimal spits in attribute values. Those attributes that are more correlated with the prediction are split on first. Deeper in the tree less relevant and irrelevant attributes are used to make prediction decisions that may only be beneficial by chance in the training dataset. This overfitting of the training data can negatively affect the modeling power of the method and cripple the predictive accuracy.

## 2. Explain what regularization is and why it is useful.
Regularization is the process of adding a tuning parameter to a model to induce smoothness in order to prevent [overfitting](https://en.wikipedia.org/wiki/Overfitting).

This is most often done by adding a constant multiple to an existing weight vector. This constant is often either the [L1 (Lasso)](https://en.wikipedia.org/wiki/Lasso_(statistics)) or [L2 (ridge)](https://en.wikipedia.org/wiki/Tikhonov_regularization), but can in actuality can be any norm. The model predictions should then minimize the mean of the loss function calculated on the regularized training set.

It is well known, as explained by others, that L1 regularization helps perform feature selection in sparse feature spaces, and that is a good practical reason to use L1 in some situations. However, beyond that particular reason I have never seen L1 to perform better than L2 in practice. If you take a look at [LIBLINEAR FAQ](https://www.csie.ntu.edu.tw/~cjlin/liblinear/FAQ.html#l1_regularized_classification) on this issue you will see how they have not seen a practical example where L1 beats L2 and encourage users of the library to contact them if they find one. Even in a situation where you might benefit from L1's sparsity in order to do feature selection, using L2 on the remaining variables is likely to give better results than L1 by itself.

## 3. What’s the difference between L1 and L2 regularization?
Regularization is a very important technique in machine learning to prevent overfitting. Mathematically speaking, it adds a regularization term in order to prevent the coefficients to fit so perfectly to overfit. The difference between the L1(Lasso) and L2(Ridge) is just that L2(Ridge) is the sum of the square of the weights, while L1(Lasso) is just the sum of the absolute weights in MSE or another loss function. As follows:
![alt text](images/regularization1.png)
The difference between their properties can be promptly summarized as follows:
![alt text](images/regularization2.png)

**Solution uniqueness** is a simpler case but requires a bit of imagination. First, this picture below:
![alt text](images/regularization3.png)

## 4. How would you validate a model you created to generate a predictive model of a quantitative outcome variable using multiple regression?
[Proposed methods](http://support.sas.com/resources/papers/proceedings12/333-2012.pdf) for model validation:
* If the values predicted by the model are far outside of the response variable range, this would immediately indicate poor estimation or model inaccuracy.
* If the values seem to be reasonable, examine the parameters; any of the following would indicate poor estimation or multi-collinearity: opposite signs of expectations, unusually large or small values, or observed inconsistency when the model is fed new data.
* Use the model for prediction by feeding it new data, and use the [coefficient of determination](https://en.wikipedia.org/wiki/Coefficient_of_determination) (R squared) as a model validity measure.
* Use data splitting to form a separate dataset for estimating model parameters, and another for validating predictions.
* Use [jackknife resampling](https://en.wikipedia.org/wiki/Jackknife_resampling) if the dataset contains a small number of instances, and measure validity with R squared and [mean squared error](https://en.wikipedia.org/wiki/Mean_squared_error) (MSE).

## 5. Explain what precision and recall are. How do they relate to the ROC curve?
Calculating precision and recall is actually quite easy. Imagine there are 100 positive cases among 10,000 cases. You want to predict which ones are positive, and you pick 200 to have a better chance of catching many of the 100 positive cases. You record the IDs of your predictions, and when you get the actual results you sum up how many times you were right or wrong. There are four ways of being right or wrong:
1. TN / True Negative: case was negative and predicted negative
2. TP / True Positive: case was positive and predicted positive
3. FN / False Negative: case was positive but predicted negative
4. FP / False Positive: case was negative but predicted positive

![alt text](images/confusion-matrix.png)

Now, your boss asks you three questions:
* What percent of your predictions were correct?
You answer: the "accuracy" was (9,760+60) out of 10,000 = 98.2%
* What percent of the positive cases did you catch?
You answer: the "recall" was 60 out of 100 = 60%
* What percent of positive predictions were correct?
You answer: the "precision" was 60 out of 200 = 30%
See also a very good explanation of [Precision and recall](https://en.wikipedia.org/wiki/Precision_and_recall) in Wikipedia.

![alt text](images/precision-recall.jpg)

ROC curve represents a relation between sensitivity (RECALL) and specificity(NOT PRECISION) and is commonly used to measure the performance of binary classifiers. However, when dealing with highly skewed datasets, [Precision-Recall (PR)](http://pages.cs.wisc.edu/~jdavis/davisgoadrichcamera2.pdf) curves give a more representative picture of performance. Remember, a ROC curve represents a relation between sensitivity (RECALL) and specificity(NOT PRECISION). Sensitivity is the other name for recall but specificity is not PRECISION.

Recall/Sensitivity is the measure of the probability that your estimate is 1 given all the samples whose true class label is 1. It is a measure of how many of the positive samples have been identified as being positive. Specificity is the measure of the probability that your estimate is 0 given all the samples whose true class label is 0. It is a measure of how many of the negative samples have been identified as being negative.

PRECISION on the other hand is different. It is a measure of the probability that a sample is a true positive class given that your classifier said it is positive. It is a measure of how many of the samples predicted by the classifier as positive is indeed positive. Note here that this changes when the base probability or prior probability of the positive class changes. Which means PRECISION depends on how rare is the positive class. In other words, it is used when positive class is more interesting than the negative class.

* Sensitivity also known as the True Positive rate or Recall is calculated as,
`Sensitivity = TP / (TP + FN)`. Since the formula doesn’t contain FP and TN, Sensitivity may give you a biased result, especially for imbalanced classes.
In the example of Fraud detection, it gives you the percentage of Correctly Predicted Frauds from the pool of Actual Frauds pool of Actual Non-Frauds.
* Specificity, also known as True Negative Rate is calculated as, `Specificity = TN / (TN + FP)`. Since the formula does not contain FN and TP, Specificity may give you a biased result, especially for imbalanced classes.
In the example of Fraud detection, it gives you the percentage of Correctly Predicted Non-Frauds from the pool of Actual Frauds pool of Actual Non-Frauds

[Assessing and Comparing Classifier Performance with ROC Curves](https://machinelearningmastery.com/assessing-comparing-classifier-performance-roc-curves-2/)

## 6. Is it better to have too many false positives, or too many false negatives?
It depends on the question as well as on the domain for which we are trying to solve the question.

In medical testing, false negatives may provide a falsely reassuring message to patients and physicians that disease is absent, when it is actually present. This sometimes leads to inappropriate or inadequate treatment of both the patient and their disease. So, it is desired to have too many false positive.

For spam filtering, a false positive occurs when spam filtering or spam blocking techniques wrongly classify a legitimate email message as spam and, as a result, interferes with its delivery. While most anti-spam tactics can block or filter a high percentage of unwanted emails, doing so without creating significant false-positive results is a much more demanding task. So, we prefer too many false negatives over many false positives.

## 7. How do you deal with unbalanced binary classification?
Imbalanced data typically refers to a problem with classification problems where the classes are not represented equally.
For example, you may have a 2-class (binary) classification problem with 100 instances (rows). A total of 80 instances are labeled with Class-1 and the remaining 20 instances are labeled with Class-2.

This is an imbalanced dataset and the ratio of Class-1 to Class-2 instances is 80:20 or more concisely 4:1.
You can have a class imbalance problem on two-class classification problems as well as multi-class classification problems. Most techniques can be used on either.
The remaining discussions will assume a two-class classification problem because it is easier to think about and describe.
1. Can You Collect More Data?</br>
A larger dataset might expose a different and perhaps more balanced perspective on the classes.
More examples of minor classes may be useful later when we look at resampling your dataset.
2. Try Changing Your Performance Metric</br>
Accuracy is not the metric to use when working with an imbalanced dataset. We have seen that it is misleading.
From that post, I recommend looking at the following performance measures that can give more insight into the accuracy of the model than traditional classification accuracy:
  - [Confusion Matrix](https://en.wikipedia.org/wiki/Confusion_matrix): A breakdown of predictions into a table showing correct predictions (the diagonal) and the types of incorrect predictions made (what classes incorrect predictions were assigned).
  - [Precision](https://en.wikipedia.org/wiki/Information_retrieval#Precision): A measure of a classifiers exactness. Precision is the number of True Positives divided by the number of True Positives and False Positives. Put another way, it is the number of positive predictions divided by the total number of positive class values predicted. It is also called the [Positive Predictive Value (PPV)](https://en.wikipedia.org/wiki/Positive_and_negative_predictive_values). Precision can be thought of as a measure of a classifiers exactness. A low precision can also indicate a large number of False Positives.
  - [Recall](https://en.wikipedia.org/wiki/Information_retrieval#Recall): A measure of a classifiers completeness. Recall is the number of True Positives divided by the number of True Positives and the number of False Negatives. Put another way it is the number of positive predictions divided by the number of positive class values in the test data. It is also called Sensitivity or the True Positive Rate. Recall can be thought of as a measure of a classifiers completeness. A low recall indicates many False Negatives.
  - [F1 Score (or F-score)](https://en.wikipedia.org/wiki/F1_score): A weighted average of precision and recall.
I would also advise you to take a look at the following:
  - Kappa (or [Cohen’s kappa](https://en.wikipedia.org/wiki/Cohen%27s_kappa)): Classification accuracy normalized by the imbalance of the classes in the data.
ROC Curves: Like precision and recall, accuracy is divided into sensitivity and specificity and models can be chosen based on the balance thresholds of these values.
3. Try Resampling Your Dataset
  * You can add copies of instances from the under-represented class called over-sampling (or more formally sampling with replacement)
  * You can delete instances from the over-represented class, called under-sampling.
5. Try Different Algorithms
6. Try Penalized Models</br>
You can use the same algorithms but give them a different perspective on the problem.
Penalized classification imposes an additional cost on the model for making classification mistakes on the minority class during training. These penalties can bias the model to pay more attention to the minority class.
Often the handling of class penalties or weights are specialized to the learning algorithm. There are penalized versions of algorithms such as penalized-SVM and penalized-LDA.
Using penalization is desirable if you are locked into a specific algorithm and are unable to resample or you’re getting poor results. It provides yet another way to “balance” the classes. Setting up the penalty matrix can be complex. You will very likely have to try a variety of penalty schemes and see what works best for your problem.
7. Try a Different Perspective</br>
Taking a look and thinking about your problem from these perspectives can sometimes shame loose some ideas.
Two you might like to consider are anomaly detection and change detection.

## 8. What is statistical power?
[Statistical power or sensitivity](https://en.wikipedia.org/wiki/Statistical_power) of a binary hypothesis test is the probability that the test correctly rejects the null hypothesis (H0) when the alternative hypothesis (H1) is true.

It can be equivalently thought of as the probability of accepting the alternative hypothesis (H1) when it is true—that is, the ability of a test to detect an effect, if the effect actually exists.

To put in another way, [Statistical power](https://effectsizefaq.com/2010/05/31/what-is-statistical-power/) is the likelihood that a study will detect an effect when the effect is present. The higher the statistical power, the less likely you are to make a Type II error (concluding there is no effect when, in fact, there is).

A type I error (or error of the first kind) is the incorrect rejection of a true null hypothesis. Usually a type I error leads one to conclude that a supposed effect or relationship exists when in fact it doesn't. Examples of type I errors include a test that shows a patient to have a disease when in fact the patient does not have the disease, a fire alarm going on indicating a fire when in fact there is no fire, or an experiment indicating that a medical treatment should cure a disease when in fact it does not.

A type II error (or error of the second kind) is the failure to reject a false null hypothesis. Examples of type II errors would be a blood test failing to detect the disease it was designed to detect, in a patient who really has the disease; a fire breaking out and the fire alarm does not ring; or a clinical trial of a medical treatment failing to show that the treatment works when really it does.
![alt text](images/statistical-power.png)

## 9. What are bias and variance, and what are their relation to modeling data?
**Bias** is how far removed a model's predictions are from correctness, while variance is the degree to which these predictions vary between model iterations.

Bias is generally the distance between the model that you build on the training data (the best model that your model space can provide) and the “real model” (which generates data).

**Error due to Bias**: Due to randomness in the underlying data sets, the resulting models will have a range of predictions. [Bias](https://en.wikipedia.org/wiki/Bias_of_an_estimator) measures how far off in general these models' predictions are from the correct value. The bias is error from erroneous assumptions in the learning algorithm. High bias can cause an algorithm to miss the relevant relations between features and target outputs (underfitting).

**Error due to Variance**: The error due to variance is taken as the variability of a model prediction for a given data point. Again, imagine you can repeat the entire model building process multiple times. The variance is how much the predictions for a given point vary between different realizations of the model. The variance is error from sensitivity to small fluctuations in the training set.

High variance can cause an algorithm to model the random [noise](https://en.wikipedia.org/wiki/Noise_(signal_processing)) in the training data, rather than the intended outputs (overfitting).

Big dataset -> low variance <br/>
Low dataset -> high variance <br/>
Few features -> high bias, low variance <br/>
Many features -> low bias, high variance <br/>
Complicated model -> low bias <br/>
Simplified model -> high bias <br/>
Decreasing λ -> low bias <br/>
Increasing λ -> low variance <br/>

We can create a graphical visualization of bias and variance using a bulls-eye diagram. Imagine that the center of the target is a model that perfectly predicts the correct values. As we move away from the bulls-eye, our predictions get worse and worse. Imagine we can repeat our entire model building process to get a number of separate hits on the target. Each hit represents an individual realization of our model, given the chance variability in the training data we gather. Sometimes we will get a good distribution of training data so we predict very well and we are close to the bulls-eye, while sometimes our training data might be full of outliers or non-standard values resulting in poorer predictions. These different realizations result in a scatter of hits on the target.
![alt text](images/bulls-eye-diagram.jpg)

[As an example](https://www.kdnuggets.com/2016/08/bias-variance-tradeoff-overview.html), using a simple flawed Presidential election survey as an example, errors in the survey are then explained through the twin lenses of bias and variance: selecting survey participants from a phonebook is a source of bias; a small sample size is a source of variance.

Minimizing total model error relies on the balancing of bias and variance errors. Ideally, models are the result of a collection of unbiased data of low variance. Unfortunately, however, the more complex a model becomes, its tendency is toward less bias but greater variance; therefore an optimal model would need to consider a balance between these 2 properties.

The statistical evaluation method of cross-validation is useful in both demonstrating the importance of this balance, as well as actually searching it out. The number of data folds to use -- the value of k in k-fold cross-validation -- is an important decision; the lower the value, the higher the bias in the error estimates and the less variance.
![alt text](images/model-complexity.jpg)

The most important takeaways are that bias and variance are two sides of an important trade-off when building models, and that even the most routine of statistical evaluation methods are directly reliant upon such a trade-off.

We may estimate a model f̂ (X) of f(X) using linear regressions or another modeling technique. In this case, the expected squared prediction error at a point x is:
`Err(x)=E[(Y−f̂ (x))^2]`

This error may then be decomposed into bias and variance components:
`Err(x)=(E[f̂ (x)]−f(x))^2+E[(f̂ (x)−E[f̂ (x)])^2]+σ^2e`
`Err(x)=Bias^2+Variance+Irreducible`

That third term, irreducible error, is the noise term in the true relationship that cannot fundamentally be reduced by any model. Given the true model and infinite data to calibrate it, we should be able to reduce both the bias and variance terms to 0. However, in a world with imperfect models and finite data, there is a tradeoff between minimizing the bias and minimizing the variance.

That third term, irreducible error, is the noise term in the true relationship that cannot fundamentally be reduced by any model. Given the true model and infinite data to calibrate it, we should be able to reduce both the bias and variance terms to 0. However, in a world with imperfect models and finite data, there is a tradeoff between minimizing the bias and minimizing the variance.

If a model is suffering from high bias, it means that model is less complex, to make the model more robust, we can add more features in feature space. Adding data points will reduce the variance.

The bias–variance tradeoff is a central problem in supervised learning. Ideally, one wants to [choose a model](https://en.wikipedia.org/wiki/Model_selection) that both accurately captures the regularities in its training data, but also generalizes well to unseen data. Unfortunately, it is typically impossible to do both simultaneously. High-variance learning methods may be able to represent their training set well, but are at risk of overfitting to noisy or unrepresentative training data. In contrast, algorithms with high bias typically produce simpler models that don't tend to overfit, but may underfit their training data, failing to capture important regularities.

Models with low bias are usually more complex (e.g. higher-order regression polynomials), enabling them to represent the training set more accurately. In the process, however, they may also represent a large noise component in the training set, making their predictions less accurate - despite their added complexity. In contrast, models with higher bias tend to be relatively simple (low-order or even linear regression polynomials), but may produce lower variance predictions when applied beyond the training set.

#### Approaches

[Dimensionality reduction](https://en.wikipedia.org/wiki/Dimensionality_reduction) and [feature selection](https://en.wikipedia.org/wiki/Feature_selection) can decrease variance by simplifying models. Similarly, a larger training set tends to decrease variance. Adding features (predictors) tends to decrease bias, at the expense of introducing additional variance. Learning algorithms typically have some tunable parameters that control bias and variance, e.g.:
* (Generalized) linear models can be [regularized](#2-explain-what-regularization-is-and-why-it-is-useful) to decrease their variance at the cost of increasing their bias.
* In artificial neural networks, the variance increases and the bias decreases with the number of hidden units. Like in GLMs, regularization is typically applied.
* In k-nearest neighbor models, a high value of k leads to high bias and low variance (see below).
* In Instance-based learning, regularization can be achieved varying the mixture of prototypes and exemplars.[
* In decision trees, the depth of the tree determines the variance. Decision trees are commonly pruned to control variance.

One way of resolving the trade-off is to use [mixture models](https://en.wikipedia.org/wiki/Mixture_model) and [ensemble learning](https://en.wikipedia.org/wiki/Ensemble_learning). For example, [boosting](https://en.wikipedia.org/wiki/Boosting_(machine_learning)) combines many "weak" (high bias) models in an ensemble that has lower bias than the individual models, while [bagging](https://en.wikipedia.org/wiki/Bootstrap_aggregating) combines "strong" learners in a way that reduces their variance.

[Understanding the Bias-Variance Tradeoff](http://scott.fortmann-roe.com/docs/BiasVariance.html)

## 10. What if the classes are imbalanced? What if there are more than 2 groups?
Binary classification involves classifying the data into two groups, e.g. whether or not a customer buys a particular product or not (Yes/No), based on independent variables such as gender, age, location etc.

As the target variable is not continuous, binary classification model predicts the probability of a target variable to be Yes/No. To evaluate such a model, a metric called the confusion matrix is used, also called the classification or co-incidence matrix. With the help of a confusion matrix, we can calculate important performance measures:
* True Positive Rate (TPR) or Recall or Sensitivity = TP / (TP + FN)
* [Precision](https://github.com/iamtodor/data-science-interview-questions-and-answers#5-explain-what-precision-and-recall-are-how-do-they-relate-to-the-roc-curve) = TP / (TP + FP)
* False Positive Rate(FPR) or False Alarm Rate = 1 - Specificity = 1 - (TN / (TN + FP))
* Accuracy = (TP + TN) / (TP + TN + FP + FN)
* Error Rate = 1 – Accuracy
* F-measure = 2 / ((1 / Precision) + (1 / Recall)) = 2 * (precision * recall) / (precision + recall)
* ROC (Receiver Operating Characteristics) = plot of FPR vs TPR
* AUC (Area Under the [ROC] Curve)  
Performance measure across all classification thresholds. Treated as the probability that a model ranks a randomly chosen positive sample higher than negative



## 11. What are some ways I can make my model more robust to outliers?
There are several ways to make a model more robust to outliers, from different points of view (data preparation or model building). An outlier in the question and answer is assumed being unwanted, unexpected, or a must-be-wrong value to the human’s knowledge so far (e.g. no one is 200 years old) rather than a rare event which is possible but rare.

Outliers are usually defined in relation to the distribution. Thus outliers could be removed in the pre-processing step (before any learning step), by using standard deviations `(Mean +/- 2*SD)`, it can be used for normality. Or interquartile ranges `Q1 - Q3`, `Q1` -  is the "middle" value in the first half of the rank-ordered data set, `Q3` - is the "middle" value in the second half of the rank-ordered data set. It can be used for not normal/unknown as threshold levels.

Moreover, data transformation (e.g. log transformation) may help if data have a noticeable tail. When outliers related to the sensitivity of the collecting instrument which may not precisely record small values, Winsorization may be useful. This type of transformation (named after Charles P. Winsor (1895–1951)) has the same effect as clipping signals (i.e. replaces extreme data values with less extreme values).  Another option to reduce the influence of outliers is using mean absolute difference rather mean squared error.

For model building, some models are resistant to outliers (e.g. tree-based approaches) or non-parametric tests. Similar to the median effect, tree models divide each node into two in each split. Thus, at each split, all data points in a bucket could be equally treated regardless of extreme values they may have.

## 12. In unsupervised learning, if a ground truth about a dataset is unknown, how can we determine the most useful number of clusters to be?
The elbow method is often the best place to state, and is especially useful due to its ease of explanation and verification via visualization. The elbow method is interested in explaining variance as a function of cluster numbers (the k in k-means). By plotting the percentage of variance explained against k, the first N clusters should add significant information, explaining variance; yet, some eventual value of k will result in a much less significant gain in information, and it is at this point that the graph will provide a noticeable angle. This angle will be the optimal number of clusters, from the perspective of the elbow method,
It should be self-evident that, in order to plot this variance against varying numbers of clusters, varying numbers of clusters must be tested. Successive complete iterations of the clustering method must be undertaken, after which the results can be plotted and compared.
DBSCAN - Density-Based Spatial Clustering of Applications with Noise. Finds core samples of high density and expands clusters from them. Good for data which contains clusters of similar density.

## 13. Define variance
Variance is the expectation of the squared deviation of a random variable from its mean. Informally, it measures how far a set of (random) numbers are spread out from their average value. The variance is the square of the standard deviation, the second central moment of a distribution, and the covariance of the random variable with itself.

Var(X) = E[(X - m)^2], m=E[X]

Variance is, thus, a measure of the scatter of the values of a random variable relative to its mathematical expectation.

## 14. Expected value
Expected value — [Expected Value](https://en.wikipedia.org/wiki/Expected_value) ([Probability Distribution](https://en.wikipedia.org/wiki/Probability_distribution) In a probability distribution, expected value is the value that a random variable takes with greatest likelihood. 

Based on the law of distribution of a random variable x, we know that a random variable x can take values x1, x2, ..., xk with probabilities p1, p2, ..., pk.
The mathematical expectation M(x) of a random variable x is equal.
The mathematical expectation of a random variable X (denoted by M (X) or less often E (X)) characterizes the average value of a random variable (discrete or continuous). Mathematical expectation is the first initial moment of a given CB.

Mathematical expectation is attributed to the so-called characteristics of the distribution position (to which the mode and median also belong). This characteristic describes a certain average position of a random variable on the numerical axis. Say, if the expectation of a random variable - the lamp life is 100 hours, then it is considered that the values of the service life are concentrated (on both sides) from this value (with dispersion on each side, indicated by the variance).

The mathematical expectation of a discrete random variable X is calculated as the sum of the products of the values xi that the CB takes X by the corresponding probabilities pi:
```python
import numpy as np
X = [3,4,5,6,7]
P = [0.1,0.2,0.3,0.4,0.5]
np.sum(np.dot(X, P))
```

## 15. Describe the differences between and use cases for box plots and histograms
A [histogram](http://www.brighthubpm.com/six-sigma/13307-what-is-a-histogram/) is a type of bar chart that graphically displays the frequencies of a data set. Similar to a bar chart, a histogram plots the frequency, or raw count, on the Y-axis (vertical) and the variable being measured on the X-axis (horizontal).

The only difference between a histogram and a bar chart is that a histogram displays frequencies for a group of data, rather than an individual data point; therefore, no spaces are present between the bars. Typically, a histogram groups data into small chunks (four to eight values per bar on the horizontal axis), unless the range of data is so great that it easier to identify general distribution trends with larger groupings.

A box plot, also called a [box-and-whisker](http://www.brighthubpm.com/six-sigma/43824-using-box-and-whiskers-plots/) plot, is a chart that graphically represents the five most important descriptive values for a data set. These values include the minimum value, the first quartile, the median, the third quartile, and the maximum value. When graphing this five-number summary, only the horizontal axis displays values. Within the quadrant, a vertical line is placed above each of the summary numbers. A box is drawn around the middle three lines (first quartile, median, and third quartile) and two lines are drawn from the box’s edges to the two endpoints (minimum and maximum).
Boxplots are better for comparing distributions than histograms!
![alt text](images/histogram-vs-boxplot.png)

## 16. How would you find an anomaly in a distribution?
Before getting started, it is important to establish some boundaries on the definition of an anomaly. Anomalies can be broadly categorized as:
1. Point anomalies: A single instance of data is anomalous if it's too far off from the rest. Business use case: Detecting credit card fraud based on "amount spent."
2. Contextual anomalies: The abnormality is context specific. This type of anomaly is common in time-series data. Business use case: Spending $100 on food every day during the holiday season is normal, but may be odd otherwise.
3. Collective anomalies: A set of data instances collectively helps in detecting anomalies. Business use case: Someone is trying to copy data form a remote machine to a local host unexpectedly, an anomaly that would be flagged as a potential cyber attack.

Best steps to prevent anomalies is to implement policies or checks that can catch them during the data collection stage. Unfortunately, you do not often get to collect your own data, and often the data you're mining was collected for another purpose. About 68% of all the data points are within one standard deviation from the mean. About 95% of the data points are within two standard deviations from the mean. Finally, over 99% of the data is within three standard deviations from the mean. When the value deviate too much from the mean, let’s say by ± 4σ, then we can considerate this almost impossible value as anomaly. (This limit can also be calculated using the percentile).

#### Statistical methods
Statistically based anomaly detection uses this knowledge to discover outliers. A dataset can be standardized by taking the z-score of each point. A z-score is a measure of how many standard deviations a data point is away from the mean of the data. Any data-point that has a z-score higher than 3 is an outlier, and likely to be an anomaly. As the z-score increases above 3, points become more obviously anomalous. A z-score is calculated using the following equation. A box-plot is perfect for this application.

#### Metric method
Judging by the number of publications, metric methods are the most popular methods among researchers. They postulate the existence of a certain metric in the space of objects, which helps to find anomalies. Intuitively, the anomaly has few neighbors in the instannce space, and a typical point has many. Therefore, a good measure of anomalies can be, for example, the «distance to the k-th neighbor». (See method: [Local Outlier Factor](https://en.wikipedia.org/wiki/Local_outlier_factor)). Specific metrics are used here, for example [Mahalonobis distance](https://en.wikipedia.org/wiki/Mahalanobis_distance). Mahalonobis distance is a measure of distance between vectors of random variables, generalizing the concept of Euclidean distance. Using Mahalonobis distance, it is possible to determine the similarity of unknown and known samples. It differs from Euclidean distance in that it takes into account correlations between variables and is scale invariant.
![alt text](images/metrical-methods.png)

The most common form of clustering-based anomaly detection is done with prototype-based clustering.

Using this approach to anomaly detection, a point is classified as an anomaly if its omission from the group significantly improves the prototype, then the point is classified as an anomaly. This logically makes sense. K-means is a clustering algorithm that clusters similar points. The points in any cluster are similar to the centroid of that cluster, hence why they are members of that cluster. If one point in the cluster is so far from the centroid that it pulls the centroid away from it's natural center, than that point is literally an outlier, since it lies outside the natural bounds for the cluster. Hence, its omission is a logical step to improve the accuracy of the rest of the cluster. Using this approach, the outlier score is defined as the degree to which a point doesn't belong to any cluster, or the distance it is from the centroid of the cluster. In K-means, the degree to which the removal of a point would increase the accuracy of the centroid is the difference in the SSE, or standard squared error, or the cluster with and without the point. If there is a substantial improvement in SSE after the removal of the point, that correlates to a high outlier score for that point.
More specifically, when using a k-means clustering approach towards anomaly detection, the outlier score is calculated in one of two ways. The simplest is the point's distance from its closest centroid. However, this approach is not as useful when there are clusters of differing densities. To tackle that problem, the point's relative distance to it's closest centroid is used, where relative distance is defined as the ratio of the point's distance from the centroid to the median distance of all points in the cluster from the centroid. This approach to anomaly detection is sensitive to the value of k. Also, if the data is highly noisy, then that will throw off the accuracy of the initial clusters, which will decrease the accuracy of this type of anomaly detection. The time complexity of this approach is obviously dependent on the choice of clustering algorithm, but since most clustering algorithms have linear or close to linear time and space complexity, this type of anomaly detection can be highly efficient.

## 17. How do you deal with outliers in your data?

For the most part, if your data is affected by these extreme cases, you can bound the input to a historical representative of your data that excludes outliers. So 
that could be a number of items (>3) or a lower or upper bounds on your order value.

If the outliers are from a data set that is relatively unique then analyze them for your specific situation. Analyze both with and without them, and perhaps with a replacement alternative, if you have a reason for one, and report your results of this assessment. 
One option is to try a transformation. Square root and log transformations both pull in high numbers.  This can make assumptions work better if the outlier is a dependent.

## 18. How do you deal with sparse data?

We could take a look at L1 regularization since it best fits to the sparse data and do feature selection. If linear relationship - linear regression either - svm. 

Also it would be nice to use one-hot-encoding or bag-of-words. A one hot encoding is a representation of categorical variables as binary vectors. This first requires that the categorical values be mapped to integer values. Then, each integer value is represented as a binary vector that is all zero values except the index of the integer, which is marked with a 1.

## 19. Big Data Engineer Can you explain what REST is?

REST stands for Representational State Transfer. (It is sometimes spelled "ReST".) It relies on a stateless, client-server, cacheable communications protocol -- and in virtually all cases, the HTTP protocol is used.
REST is an architecture style for designing networked applications. The idea is simple HTTP is used to make calls between machines.
* In many ways, the World Wide Web itself, based on HTTP, can be viewed as a REST-based architecture.
RESTful applications use HTTP requests to post data (create and/or update), read data (e.g., make queries), and delete data. Thus, REST uses HTTP for all four CRUD (Create/Read/Update/Delete) operations.
REST is a lightweight alternative to mechanisms like RPC (Remote Procedure Calls) and Web Services (SOAP, WSDL, et al.). Later, we will see how much more simple REST is.
* Despite being simple, REST is fully-featured; there's basically nothing you can do in Web Services that can't be done with a RESTful architecture.
REST is not a "standard". There will never be a W3C recommendation for REST, for example. And while there are REST programming frameworks, working with REST is so simple that you can often "roll your own" with standard library features in languages like Perl, Java, or C#.

## 20. Logistic regression

Log odds - raw output from the model; odds - exponent from the output of the model. Probability of the output - odds / (1+odds).

## 21. What is the effect on the coefficients of logistic regression if two predictors are highly correlated? What are the confidence intervals of the coefficients?
When predictor variables are correlated, the estimated regression coefficient of any one variable depends on which other predictor variables are included in the model. When predictor variables are correlated, the precision of the estimated regression coefficients decreases as more predictor variables are added to the model.

In statistics, multicollinearity (also collinearity) is a phenomenon in which two or more predictor variables in a multiple regression model are highly correlated, meaning that one can be linearly predicted from the others with a substantial degree of accuracy. In this situation the coefficient estimates of the multiple regression may change erratically in response to small changes in the model or the data. Multicollinearity does not reduce the predictive power or reliability of the model as a whole, at least within the sample data set; it only affects calculations regarding individual predictors. That is, a multiple regression model with correlated predictors can indicate how well the entire bundle of predictors predicts the outcome variable, but it may not give valid results about any individual predictor, or about which predictors are redundant with respect to others.

The consequences of multicollinearity:
* Ratings estimates remain unbiased.
* Standard coefficient errors increase.
* The calculated t-statistics are underestimated.
* Estimates become very sensitive to changes in specifications and changes in individual observations.
* The overall quality of the equation, as well as estimates of variables not related to multicollinearity, remain unaffected.
* The closer multicollinearity to perfect (strict), the more serious its consequences.

Indicators of multicollinearity: 
1. High R2 and negligible odds.
2. Strong pair correlation of predictors.
3. Strong partial correlations of predictors.
4. High VIF - variance inflation factor.

Confidence interval (CI) is a type of interval estimate (of a population parameter) that is computed from the observed data. The confidence level is the frequency (i.e., the proportion) of possible confidence intervals that contain the true value of their corresponding parameter. In other words, if confidence intervals are constructed using a given confidence level in an infinite number of independent experiments, the proportion of those intervals that contain the true value of the parameter will match the confidence level.

Confidence intervals consist of a range of values (interval) that act as good estimates of the unknown population parameter. However, the interval computed from a particular sample does not necessarily include the true value of the parameter. Since the observed data are random samples from the true population, the confidence interval obtained from the data is also random. If a corresponding hypothesis test is performed, the confidence level is the complement of the level of significance, i.e. a 95% confidence interval reflects a significance level of 0.05. If it is hypothesized that a true parameter value is 0 but the 95% confidence interval does not contain 0, then the estimate is significantly different from zero at the 5% significance level.

The desired level of confidence is set by the researcher (not determined by data). Most commonly, the 95% confidence level is used. However, other confidence levels can be used, for example, 90% and 99%.

Factors affecting the width of the confidence interval include the size of the sample, the confidence level, and the variability in the sample. A larger sample size normally will lead to a better estimate of the population parameter.
A Confidence Interval is a range of values we are fairly sure our true value lies in.

`X  ±  Z*s/√(n)`, X is the mean, Z is the chosen Z-value from the table, s is the standard deviation, n is the number of samples. The value after the ± is called the margin of error.

## 22. What’s the difference between Gaussian Mixture Model and K-Means?
Let's says we are aiming to break them into three clusters. K-means will start with the assumption that a given data point belongs to one cluster.

Choose a data point. At a given point in the algorithm, we are certain that a point belongs to a red cluster. In the next iteration, we might revise that belief, and be certain that it belongs to the green cluster. However, remember, in each iteration, we are absolutely certain as to which cluster the point belongs to. This is the "hard assignment".

What if we are uncertain? What if we think, well, I can't be sure, but there is 70% chance it belongs to the red cluster, but also 10% chance its in green, 20% chance it might be blue. That's a soft assignment. The Mixture of Gaussian model helps us to express this uncertainty. It starts with some prior belief about how certain we are about each point's cluster assignments. As it goes on, it revises those beliefs. But it incorporates the degree of uncertainty we have about our assignment.

Kmeans: find kk to minimize `(x−μk)^2`

Gaussian Mixture (EM clustering) : find kk to minimize `(x−μk)^2/σ^2`

The difference (mathematically) is the denominator “σ^2”, which means GM takes variance into consideration when it calculates the measurement.
Kmeans only calculates conventional Euclidean distance.
In other words, Kmeans calculate distance, while GM calculates “weighted” distance.

**K means**:
* Hard assign a data point to one particular cluster on convergence.
* It makes use of the L2 norm when optimizing (Min {Theta} L2 norm point and its centroid coordinates).

**EM**:
* Soft assigns a point to clusters (so it give a probability of any point belonging to any centroid).
* It doesn't depend on the L2 norm, but is based on the Expectation, i.e., the probability of the point belonging to a particular cluster. This makes K-means biased towards spherical clusters.

## 23. Describe how Gradient Boosting works.
The idea of boosting came out of the idea of whether a weak learner can be modified to become better.

Gradient boosting relies on regression trees (even when solving a classification problem) which minimize **MSE**. Selecting a prediction for a leaf region is simple: to minimize MSE we should select an average target value over samples in the leaf. The tree is built greedily starting from the root: for each leaf a split is selected to minimize MSE for this step.

To begin with, gradient boosting is an ensembling technique, which means that prediction is done by an ensemble of simpler estimators. While this theoretical framework makes it possible to create an ensemble of various estimators, in practice we almost always use GBDT — gradient boosting over decision trees. 

The aim of gradient boosting is to create (or "train") an ensemble of trees, given that we know how to train a single decision tree. This technique is called **boosting** because we expect an ensemble to work much better than a single estimator.

Here comes the most interesting part. Gradient boosting builds an ensemble of trees **one-by-one**, then the predictions of the individual trees **are summed**: D(x)=d​tree 1​​(x)+d​tree 2​​(x)+...

The next decision tree tries to cover the discrepancy between the target function f(x) and the current ensemble prediction **by reconstructing the residual**.

For example, if an ensemble has 3 trees the prediction of that ensemble is:
D(x)=d​tree 1​​(x)+d​tree 2​​(x)+d​tree 3​​(x). The next tree (tree 4) in the ensemble should complement well the existing trees and minimize the training error of the ensemble.

In the ideal case we'd be happy to have: D(x)+d​tree 4​​(x)=f(x).

To get a bit closer to the destination, we train a tree to reconstruct the difference between the target function and the current predictions of an ensemble, which is called the **residual**: R(x)=f(x)−D(x). Did you notice? If decision tree completely reconstructs R(x), the whole ensemble gives predictions without errors (after adding the newly-trained tree to the ensemble)! That said, in practice this never happens, so we instead continue the iterative process of ensemble building.

### AdaBoost the First Boosting Algorithm
The weak learners in AdaBoost are decision trees with a single split, called decision stumps for their shortness.

AdaBoost works by weighting the observations, putting more weight on difficult to classify instances and less on those already handled well. New weak learners are added sequentially that focus their training on the more difficult patterns.
**Gradient boosting involves three elements:**
1. A loss function to be optimized.
2. A weak learner to make predictions.
3. An additive model to add weak learners to minimize the loss function.

#### Loss Function
The loss function used depends on the type of problem being solved.
It must be differentiable, but many standard loss functions are supported and you can define your own.
For example, regression may use a squared error and classification may use logarithmic loss.
A benefit of the gradient boosting framework is that a new boosting algorithm does not have to be derived for each loss function that may want to be used, instead, it is a generic enough framework that any differentiable loss function can be used.

#### Weak Learner
Decision trees are used as the weak learner in gradient boosting.

Specifically regression trees are used that output real values for splits and whose output can be added together, allowing subsequent models outputs to be added and “correct” the residuals in the predictions.

Trees are constructed in a greedy manner, choosing the best split points based on purity scores like Gini or to minimize the loss.
Initially, such as in the case of AdaBoost, very short decision trees were used that only had a single split, called a decision stump. Larger trees can be used generally with 4-to-8 levels.

It is common to constrain the weak learners in specific ways, such as a maximum number of layers, nodes, splits or leaf nodes.
This is to ensure that the learners remain weak, but can still be constructed in a greedy manner.

#### Additive Model
Trees are added one at a time, and existing trees in the model are not changed.

A gradient descent procedure is used to minimize the loss when adding trees.
Traditionally, gradient descent is used to minimize a set of parameters, such as the coefficients in a regression equation or weights in a neural network. After calculating error or loss, the weights are updated to minimize that error.

Instead of parameters, we have weak learner sub-models or more specifically decision trees. After calculating the loss, to perform the gradient descent procedure, we must add a tree to the model that reduces the loss (i.e. follow the gradient). We do this by parameterizing the tree, then modify the parameters of the tree and move in the right direction by reducing the residual loss.

Generally this approach is called functional gradient descent or gradient descent with functions.
The output for the new tree is then added to the output of the existing sequence of trees in an effort to correct or improve the final output of the model.

A fixed number of trees are added or training stops once loss reaches an acceptable level or no longer improves on an external validation dataset.

### Improvements to Basic Gradient Boosting
Gradient boosting is a greedy algorithm and can overfit a training dataset quickly.
It can benefit from regularization methods that penalize various parts of the algorithm and generally improve the performance of the algorithm by reducing overfitting.
In this section we will look at 4 enhancements to basic gradient boosting:
* Tree Constraints
* Shrinkage
* Random sampling
* Penalized Learning

#### Tree Constraints
It is important that the weak learners have skill but remain weak.
There are a number of ways that the trees can be constrained.

A good general heuristic is that the more constrained tree creation is, the more trees you will need in the model, and the reverse, where less constrained individual trees, the fewer trees that will be required.

Below are some constraints that can be imposed on the construction of decision trees:
* Number of trees, generally adding more trees to the model can be very slow to overfit. The advice is to keep adding trees until no further improvement is observed.
* Tree depth, deeper trees are more complex trees and shorter trees are preferred. Generally, better results are seen with 4-8 levels.
* Number of nodes or number of leaves, like depth, this can constrain the size of the tree, but is not constrained to a symmetrical structure if other constraints are used.
* Number of observations per split imposes a minimum constraint on the amount of training data at a training node before a split can be considered
* Minimum improvement to loss is a constraint on the improvement of any split added to a tree.

#### Weighted Updates
The predictions of each tree are added together sequentially.
The contribution of each tree to this sum can be weighted to slow down the learning by the algorithm. This weighting is called a shrinkage or a learning rate.

Each update is simply scaled by the value of the “learning rate parameter” *v*

The effect is that learning is slowed down, in turn require more trees to be added to the model, in turn taking longer to train, providing a configuration trade-off between the number of trees and learning rate.

Decreasing the value of v [the learning rate] increases the best value for M [the number of trees].

It is common to have small values in the range of 0.1 to 0.3, as well as values less than 0.1.

Similar to a learning rate in stochastic optimization, shrinkage reduces the influence of each individual tree and leaves space for future trees to improve the model.
#### Stochastic Gradient Boosting
A big insight into bagging ensembles and random forest was allowing trees to be greedily created from subsamples of the training dataset.

This same benefit can be used to reduce the correlation between the trees in the sequence in gradient boosting models.

This variation of boosting is called stochastic gradient boosting.

At each iteration a subsample of the training data is drawn at random (without replacement) from the full training dataset. The randomly selected subsample is then used, instead of the full sample, to fit the base learner.

A few variants of stochastic boosting that can be used:
* Subsample rows before creating each tree.
* Subsample columns before creating each tree
* Subsample columns before considering each split.
Generally, aggressive sub-sampling such as selecting only 50% of the data has shown to be beneficial. According to user feedback, using column sub-sampling prevents over-fitting even more so than the traditional row sub-sampling.
#### Penalized Gradient Boosting
Additional constraints can be imposed on the parameterized trees in addition to their structure.
Classical decision trees like CART are not used as weak learners, instead a modified form called a regression tree is used that has numeric values in the leaf nodes (also called terminal nodes). The values in the leaves of the trees can be called weights in some literature.

As such, the leaf weight values of the trees can be regularized using popular regularization functions, such as:
* L1 regularization of weights.
* L2 regularization of weights.

The additional regularization term helps to smooth the final learnt weights to avoid over-fitting. Intuitively, the regularized objective will tend to select a model employing simple and predictive functions.

More details in 2 posts (russian):
* https://habr.com/company/ods/blog/327250/
* https://alexanderdyakonov.files.wordpress.com/2017/06/book_boosting_pdf.pdf

## 24. Difference between AdaBoost and XGBoost.
Both methods combine weak learners into one strong learner. For example, one decision tree is a weak learner, and an emsemble of them would be a random forest model, which is a strong learner. 

Both methods in the learning process will increase the ensemble of weak-trainers, adding new weak learners to the ensemble at each training iteration, i.e. in the case of the forest, the forest will grow with new trees. The only difference between AdaBoost and XGBoost is how the ensemble is replenished.

AdaBoost works by weighting the observations, putting more weight on difficult to classify instances and less on those already handled well. New weak learners are added sequentially that focus their training on the more difficult patterns. AdaBoost at each iteration changes the sample weights in the sample. It raises the weight of the samples in which more mistakes were made. The sample weights vary in proportion to the ensemble error. We thereby change the probabilistic distribution of samples - those that have more weight will be selected more often in the future. It is as if we had accumulated samples on which more mistakes were made and would use them instead of the original sample. In addition, in AdaBoost, each weak learner has its own weight in the ensemble (alpha weight) - this weight is higher, the “smarter” this weak learner is, i.e. than the learner least likely to make mistakes.

XGBoost does not change the selection or the distribution of observations at all. XGBoost builds the first tree (weak learner), which will fit the observations with some prediction error. A second tree (weak learner) is then added to correct the errors made by the existing model. Errors are minimized using a gradient descent algorithm. Regularization can also be used to penalize more complex models through both Lasso and Ridge regularization. 

In short, AdaBoost- reweighting examples. Gradient boosting - predicting the loss function of trees. Xgboost - the regularization term was added to the loss function (depth + values ​​in leaves).

## 25. Data Mining Describe the decision tree model
A decision tree is a structure that includes a root node, branches, and leaf nodes. Each internal node denotes a test on an attribute, each branch denotes the outcome of a test, and each leaf node holds a class label. The topmost node in the tree is the root node.

Each internal node represents a test on an attribute. Each leaf node represents a class.
The benefits of having a decision tree are as follows:
* It does not require any domain knowledge.
* It is easy to comprehend.
* The learning and classification steps of a decision tree are simple and fast.

**Tree Pruning**

Tree pruning is performed in order to remove anomalies in the training data due to noise or outliers. The pruned trees are smaller and less complex.

**Tree Pruning Approaches**

Here is the Tree Pruning Approaches listed below:
* Pre-pruning − The tree is pruned by halting its construction early.
* Post-pruning - This approach removes a sub-tree from a fully grown tree.

**Cost Complexity**

The cost complexity is measured by the following two parameters − Number of leaves in the tree, and Error rate of the tree.

## 26. Notes from Coursera Deep Learning courses by Andrew Ng
[Notes from Coursera Deep Learning courses by Andrew Ng](https://pt.slideshare.net/TessFerrandez/notes-from-coursera-deep-learning-courses-by-andrew-ng/)

## 27. What is a neural network?
Neural networks are typically organized in layers. Layers are made up of a number of interconnected 'nodes' which contain an 'activation function'. Patterns are presented to the network via the 'input layer', which communicates to one or more 'hidden layers' where the actual processing is done via a system of weighted 'connections'. The hidden layers then link to an 'output layer' where the answer is output as shown in the graphic below.

Although there are many different kinds of learning rules used by neural networks, this demonstration is concerned only with one: the delta rule. The delta rule is often utilized by the most common class of ANNs called 'backpropagation neural networks' (BPNNs). Backpropagation is an abbreviation for the backwards propagation of error. With the delta rule, as with other types of back propagation, 'learning' is a supervised process that occurs with each cycle or 'epoch' (i.e. each time the network is presented with a new input pattern) through a forward activation flow of outputs, and the backwards error propagation of weight adjustments. More simply, when a neural network is initially presented with a pattern it makes a random 'guess' as to what it might be. It then sees how far its answer was from the actual one and makes an appropriate adjustment to its connection weights. More graphically, the process looks something like this: 
![alt text](images/neural_network.png)

Backpropagation performs a gradient descent within the solution's vector space towards a 'global minimum' along the steepest vector of the error surface. The global minimum is that theoretical solution with the lowest possible error. The error surface itself is a hyperparaboloid but is seldom 'smooth'. Indeed, in most problems, the solution space is quite irregular with numerous 'pits' and 'hills' which may cause the network to settle down in a 'local minimum' which is not the best overall solution.

Since the nature of the error space can not be known a priori, neural network analysis often requires a large number of individual runs to determine the best solution. Most learning rules have built-in mathematical terms to assist in this process which control the 'speed' (Beta-coefficient) and the 'momentum' of the learning. The speed of learning is actually the rate of convergence between the current solution and the global minimum. Momentum helps the network to overcome obstacles (local minima) in the error surface and settle down at or near the global minimum.

Once a neural network is 'trained' to a satisfactory level it may be used as an analytical tool on other data. To do this, the user no longer specifies any training runs and instead allows the network to work in forward propagation mode only. New inputs are presented to the input pattern where they filter into and are processed by the middle layers as though training were taking place, however, at this point the output is retained and no backpropagation occurs. The output of a forward propagation run is the predicted model for the data which can then be used for further analysis and interpretation.

## 28. How do you deal with sparse data?
We could take a look at L1 regularization since it best fits the sparse data and does feature selection. If linear relationship - linear regression either - svm. Also it would be nice to use one-hot-encoding or bag-of-words. 
A one hot encoding is a representation of categorical variables as binary vectors.
This first requires that the categorical values be mapped to integer values.
Then, each integer value is represented as a binary vector that is all zero values except the index of the integer, which is marked with a 1.

## 29. RNN and LSTM
Here are a few of my favorites:
* [Understanding LSTM Networks, Chris Olah's LSTM post](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
* [Exploring LSTMs, Edwin Chen's LSTM post](http://blog.echen.me/2017/05/30/exploring-lstms/)
* [The Unreasonable Effectiveness of Recurrent Neural Networks, Andrej Karpathy's blog post](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
* [CS231n Lecture 10 - Recurrent Neural Networks, Image Captioning, LSTM, Andrej Karpathy's lecture](https://www.youtube.com/watch?v=iX5V1WpxxkY)
* [Jay Alammar's The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) the guy generally focuses on visualizing different ML concepts

## 30. Pseudo Labeling
Pseudo-labeling is a technique that allows you to use predicted with **confidence** test data in your training process. This effectivey works by allowing your model to look at more samples, possibly varying in distributions. I have found [this](https://www.kaggle.com/cdeotte/pseudo-labeling-qda-0-969) Kaggle kernel to be useful in understanding how one can use pseudo-labeling in light of having too few train data points.

## 31. Knowledge Distillation
It is the process by which a considerably larger model is able to transfer its knowledge to a smaller one. Applications include NLP and object detection allowing for less powerful hardware to make good inferences without significant loss of accuracy.

Example: model compression which is used to compress the knowledge of multiple models into a single neural network.

[Explanation](https://nervanasystems.github.io/distiller/knowledge_distillation.html)

## 32. What is an inductive bias?
A model's inductive bias is referred to as assumptions made within that model to learn your target function from independent variables, your features. Without these assumptions, there is a whole space of solutions to our problem and finding the one that works best becomes a problem. Found [this](https://stackoverflow.com/questions/35655267/what-is-inductive-bias-in-machine-learning) StackOverflow question useful to look at and explore.

Consider an example of an inducion bias when choosing a learning algorithm with the minimum cross-validation (CV) error. Here, we **rely** on the hypothesis of the minimum CV error and **hope** it is able to generalize well on the data yet to be seen. Effectively, this choice is what helps us (in this case) make a choice in favor of the learning algorithm (or model) being tried.

## 33. What is a confidence interval in layman's terms?
Confidence interval as the name suggests is the amount of confidence associated with an interval of values to get the desired outcome. For example : if 100 - 200 range is a 95% confidence interval , it implies that someone can have 95% assurance that the data point or any desired value is present in that range.



Data-Science-Interview-Questions-and-Answers-General (Updating)
====================================================

I hope this article could help beginners to better understanding of Data Science, and have a better performance in your first interviews.  

I will do long update and please feel free to contact me if you have any questions.  

I'm just a porter, most of them are borrowing from others

## Data Science Questions and Answers (General) for beginner
### Editor : Zhiqiang ZHONG 

# Content
#### Q1 How would you create a taxonomy to identify key customer trends in unstructured data?

    The best way to approach this question is to mention that it is good to check with the business owner 
    and understand their objectives before categorizing the data. Having done this, it is always good to 
    follow an iterative approach by pulling new data samples and improving   the model accordingly by validating 
    it for accuracy by soliciting feedback from the stakeholders of the business. This helps ensure that your 
    model is producing actionable results and improving over the time.
    
#### Q2 Python or R – Which one would you prefer for text analytics?

    The best possible answer for this would be Python because it has Pandas library that provides easy to use 
    data structures and high performance data analysis tools.
    
#### Q3 Which technique is used to predict categorical responses?

    Classification technique is used widely in mining for classifying data sets.
    
#### Q4 What is logistic regression? Or State an example when you have used logistic regression recently.

    Logistic Regression often referred as logit model is a technique to predict the binary outcome from a linear 
    combination of predictor variables. For example, if you want to predict whether a particular political leader 
    will win the election or not. In this case, the outcome of prediction is binary i.e. 0 or 1 (Win/Lose). The 
    predictor variables here would be the amount of money spent for election campaigning of a particular candidate, 
    the amount of time spent in campaigning, etc.
    
#### Q5 What are Recommender Systems?

    A subclass of information filtering systems that are meant to predict the preferences or ratings that a user 
    would give to a product. Recommender systems are widely used in movies, news, research articles, products, 
    social tags, music, etc.
    
#### Q6 Why data cleaning plays a vital role in analysis?

    Cleaning data from multiple sources to transform it into a format that data analysts or data scientists can work 
    with is a cumbersome process because - as the number of data sources increases, the time take to clean the data 
    increases exponentially due to the number of sources and the volume of data generated in these sources. It might 
    take up to 80% of the time for just cleaning data making it a critical part of analysis task.
    
#### Q7 Differentiate between univariate, bivariate and multivariate analysis.

    These are descriptive statistical analysis techniques which can be differentiated based on the number of 
    variables involved at a given point of time. For example, the pie charts of sales based on territory involve 
    only one variable and can be referred to as univariate analysis.

    If the analysis attempts to understand the difference between 2 variables at time as in a scatterplot, then it 
    is referred to as bivariate analysis. For example, analysing the volume of sale and a spending can be considered 
    as an example of bivariate analysis.

    Analysis that deals with the study of more than two variables to understand the effect of variables on the 
    responses is referred to as multivariate analysis.

#### Q8 What do you understand by the term Normal Distribution?

    Data is usually distributed in different ways with a bias to the left or to the right or it can all be jumbled
    up. However, there are chances that data is distributed around a central value without any bias to the left or
    right and reaches normal distribution in the form of a bell shaped curve. The random variables are distributed
    in the form of an symmetrical bell shaped curve.
    
![](https://s3.amazonaws.com/files.dezyre.com/images/blog/100+Data+Science+Interview+Questions+and+Answers+(General)/Bell+Shaped+Curve+for+Normal+Distribution.jpg)

#### Q9 What is Linear Regression?

    Linear regression is a statistical technique where the score of a variable Y is predicted from the score of a 
    second variable X. X is referred to as the predictor variable and Y as the criterion variable.
    
#### Q10 What is Interpolation and Extrapolation?

    Estimating a value from 2 known values from a list of values is Interpolation. Extrapolation is approximating 
    a value by extending a known set of values or facts.
    
#### Q11 What is power analysis?

    An experimental design technique for determining the effect of a given sample size.
    
#### Q12 What is K-means? How can you select K for K-means?

    K-means is a clestering algorithm, handle with un-supervised problem. k-means clustering aims to partition
    n observations into k clusters in which each observation belongs to the cluster with the nearest mean, 
    serving as a prototype of the cluster.
    
    You can choose the number of cluster by visually but there is lots of ambiguity, or computethe sum of SSE(the
    sum of squared error) for some values of K. To find one good K.
    
![](https://qph.ec.quoracdn.net/main-qimg-678795190794dd4c071366c06bf32115.webp)

    In this case, k=6 is the value.
    
[More reading](https://www.quora.com/How-can-we-choose-a-good-K-for-K-means-clustering)
    
#### Q13 What is Collaborative filtering?

    The process of filtering used by most of the recommender systems to find patterns or information by collaborating 
    viewpoints, various data sources and multiple agents.
    
#### Q14 What is the difference between Cluster and Systematic Sampling?

    Cluster sampling is a technique used when it becomes difficult to study the target population spread across
    a wide area and simple random sampling cannot be applied. Cluster Sample is a probability sample where each 
    sampling unit is a collection, or cluster of elements. Systematic sampling is a statistical technique where 
    elements are selected from an ordered sampling frame. In systematic sampling, the list is progressed in a 
    circular manner so once you reach the end of the list,it is progressed from the top again. The best example
    for systematic sampling is equal probability method.
    
#### Q15 Are expected value and mean value different?

    They are not different but the terms are used in different contexts. Mean is generally referred when talking 
    about a probability distribution or sample population whereas expected value is generally referred in a 
    random variable context.

    ***For Sampling Data***
    Mean value is the only value that comes from the sampling data.
    Expected Value is the mean of all the means i.e. the value that is built from multiple samples. Expected 
    value is the population mean.

    ***For Distributions***
    Mean value and Expected value are same irrespective of the distribution, under the condition that the 
    distribution is in the same population.
    
#### Q16 What does P-value signify about the statistical data?

    P-value is used to determine the significance of results after a hypothesis test in statistics. P-value 
    helps the readers to draw conclusions and is always between 0 and 1.
- P- Value > 0.05 denotes weak evidence against the null hypothesis which means the null hypothesis cannot be rejected.
- P-value <= 0.05 denotes strong evidence against the null hypothesis which means the null hypothesis can be rejected.
- P-value=0.05is the marginal value indicating it is possible to go either way.

#### Q17 Do gradient descent methods always converge to same point?

    No, they do not because in some cases it reaches a local minima or a local optima point. You don’t reach 
    the global optima point. It depends on the data and starting conditions
    
~~#### Q18 What are categorical variables?~~

#### Q19 A test has a true positive rate of 100% and false positive rate of 5%. There is a population with a 1/1000 rate of having the condition the test identifies. Considering a positive test, what is the probability of having that condition?

    Let’s suppose you are being tested for a disease, if you have the illness the test will end up saying you 
    have the illness. However, if you don’t have the illness- 5% of the times the test will end up saying you
    have the illness and 95% of the times the test will give accurate result that you don’t have the illness. 
    Thus there is a 5% error in case you do not have the illness.

    Out of 1000 people, 1 person who has the disease will get true positive result.

    Out of the remaining 999 people, 5% will also get true positive result.

    Close to 50 people will get a true positive result for the disease.

    This means that out of 1000 people, 51 people will be tested positive for the disease even though only one 
    person has the illness. There is only a 2% probability of you having the disease even if your reports say 
    that you have the disease.

#### Q20 How you can make data normal using Box-Cox transformation?

    The calculation fomula of Box-Cox: 
![](http://images.cnblogs.com/cnblogs_com/zgw21cn/WindowsLiveWriter/BoxCox_119E9/clip_image002_thumb.gif)

    It change the calculation between log, sqrt and reciprocal operation by changing lambda. Find a suitable 
    lambda based on specific data set.
    
#### Q21 What is the difference between Supervised Learning an Unsupervised Learning?

    If an algorithm learns something from the training data so that the knowledge can be applied to the test data,
    then it is referred to as Supervised Learning. Classification is an example for Supervised Learning. If the
    algorithm does not learn anything beforehand because there is no response variable or any training data, 
    then it is referred to as unsupervised learning. Clustering is an example for unsupervised learning.
    
#### Q22 Explain the use of Combinatorics in data science.

    Combinatorics used a lot in data science, from feature engineer to algorithms(ensemble algorithms).Creat new features
    by merge original feature and merge several networks in one to creat news, like bagging, boosting and stacking. 

#### Q23 Why is vectorization considered a powerful method for optimizing numerical code?

    Vectorization can change original data to be structed.

#### Q24 What is the goal of A/B Testing?

    It is a statistical hypothesis testing for randomized experiment with two variables A and B. The goal of A/B 
    Testing is to identify any changes to the web page to maximize or increase the outcome of an interest. An
    example for this could be identifying the click through rate for a banner ad.
    
#### Q25 What is an Eigenvalue and Eigenvector?

    Eigenvectors are used for understanding linear transformations. In data analysis, we usually calculate the
    eigenvectors for a correlation or covariance matrix. Eigenvectors are the directions along which a particular
    linear transformation acts by flipping, compressing or stretching. Eigenvalue can be referred to as the strength
    of the transformation in the direction of eigenvector or the factor by which the compression occurs.
#### Q26 What is Gradient Descent?

    A method to find the local minimum of a function. From a point along the direction of gradient to iterational 
    search by a certain step length, until gradient equals zero. 

#### Q27 How can outlier values be treated?

    Outlier values can be identified by using univariate or any other graphical analysis method. If the number of
    outlier values is few then they can be assessed individually but for large number of outliers the values can
    be substituted with either the 99th or the 1st percentile values. All extreme values are not outlier values.
    The most common ways to treat outlier values –
    
1. To change the value and bring in within a range

2. To just remove the value.

#### Q28 How can you assess a good logistic model?

    There are various methods to assess the results of a logistic regression analysis-
    
- Using Classification Matrix to look at the true negatives and false positives.
- Concordance that helps identify the ability of the logistic model to differentiate between the event happening and not happening.
- Lift helps assess the logistic model by comparing it with random selection.

#### Q29 What are various steps involved in an analytics project?

- Understand the business problem
- Explore the data and become familiar with it.
- Prepare the data for modelling by detecting outliers, treating missing values, transforming variables, etc.
- After data preparation, start running the model, analyse the result and tweak the approach. This is an iterative step till the best possible outcome is achieved.
- Validate the model using a new data set.
- Start implementing the model and track the result to analyse the performance of the model over the period of time.

#### Q30 How can you iterate over a list and also retrieve element indices at the same time?

    This can be done using the enumerate function which takes every element in a sequence just like in a list
    and adds its location just before it.
    
#### Q31 During analysis, how do you treat missing values?

Minsing values has many reasons, like:
- Information not advisable for this time
- Information was missed by collect
- Some attributes of some items are not avaliable
- Some information was thinked not important
- It's too expensive to collect all these data
    
Types of Missing values:
- Missing completely at Random (MCAR): no relationship with missing values and other variables, like 
    family adress
- Missing at random (MAR): not completely random, missing denpends on other variables, like finance situation
    data missing has relationship with the company size
- Missing not at random (MNAR): there is relationship with the value of variable self, like high income families 
    don't will to open its income situation
      
Methods treatment (you need to know clearly about your missing values firstly)
- Delect tuple
    Delect tuples have any missing values
    - List wise delection
    - Pair wise delection
![](https://www.analyticsvidhya.com/wp-content/uploads/2015/02/Data_Exploration_2_2.png)

- Imputation
    - Filling manually
    - Treating Missing Attribute values as Special values (mean, mode, median imputation)
    - Hot deck imputation
    - KNN 
    - Assigning All Possible values of the Attribute
    - Combinational Completer
    - Regression
    - Expectation maximization, EM
    - Multiple Imputation

[More Reading (In Chinese)](http://blog.csdn.net/lujiandong1/article/details/52654703)

[Python package](https://pypi.python.org/pypi/fancyimpute)

~~#### Q32 Explain about the box cox transformation in regression models.~~

#### Q33 Can you use machine learning for time series analysis?

    Yes, it can be used but it depends on the applications.
    
#### Q34 Write a function that takes in two sorted lists and outputs a sorted list that is their union. 

    First solution which will come to your mind is to merge two lists and short them afterwards
    **Python code-**
    def return_union(list_a, list_b):
        return sorted(list_a + list_b)
    
    **R code-**
    return_union <- function(list_a, list_b)
    {
    list_c<-list(c(unlist(list_a),unlist(list_b)))
    return(list(list_c[[1]][order(list_c[[1]])]))
    }

    Generally, the tricky part of the question is not to use any sorting or ordering function. In that 
    case you will have to write your own logic to answer the question and impress your interviewer.
    
    ***Python code-***
    def return_union(list_a, list_b):
        len1 = len(list_a)
        len2 = len(list_b)
        final_sorted_list = []
        j = 0
        k = 0
    
        for i in range(len1+len2):
            if k == len1:
                final_sorted_list.extend(list_b[j:])
                break
            elif j == len2:
                final_sorted_list.extend(list_a[k:])
                break
            elif list_a[k] < list_b[j]:
                final_sorted_list.append(list_a[k])
                k += 1
            else:
                final_sorted_list.append(list_b[j])
                j += 1
        return final_sorted_list

    Similar function can be returned in R as well by following the similar steps.

    return_union <- function(list_a,list_b)
    {
    #Initializing length variables
    len_a <- length(list_a)
    len_b <- length(list_b)
    len <- len_a + len_b
    
    #initializing counter variables
    
    j=1
    k=1
    
    #Creating an empty list which has length equal to sum of both the lists
    
    list_c <- list(rep(NA,len))
    
    #Here goes our for loop 
    
    for(i in 1:len)
    {
        if(j>len_a)
        {
            list_c[i:len] <- list_b[k:len_b]
            break
        }
        else if(k>len_b)
        {
            list_c[i:len] <- list_a[j:len_a]
            break
        }
        else if(list_a[[j]] <= list_b[[k]])
        {
            list_c[[i]] <- list_a[[j]]
            j <- j+1
        }
        else if(list_a[[j]] > list_b[[k]])
        {
        list_c[[i]] <- list_b[[k]]
        k <- k+1
        }
    }
    return(list(unlist(list_c)))

    }
#### Q35 What is the difference between Bayesian Inference and Maximum Likelihood Estimation (MLE)?

#### Q36 What is Regularization and what kind of problems does regularization solve?
    A central problem in machine learning is how to make an algorithm that will perform weel not just on
    the training data, but also on new inputs. Many strategies used in machine learning are explicitly 
    designed to reduce the test error, possibly at the expense of increased training error. These 
    strategies are known collectively as regularization.
    Briefly, regularization is any modification we make to a learning algorithm that is intended to 
    reduce its generalization error but not its training error.

#### Q37 What is multicollinearity and how you can overcome it?
    In statistics, multicollinearity (also collinearity) is a phenomenon in which two or more predictor
    variables in a multiple regression model are highly correlated, meaning that one can be linearly 
    predicted from the others with a substantial degree of accuracy. 
    Solutions:
        Remove variables that lead to multicollinearity.
        Obtain more data.
        Ridge regression or PCA (principal component regression) or partial least squares regression
[More reading in WIKI](https://en.wikipedia.org/wiki/Multicollinearity)

#### Q38 What is the curse of dimensionality?
    It refers to various phenomena that arise when analyzing and organizing data in high-dimensional 
    spaces (often with hundreds or thousands of dimensions) that do not occur in low-dimensional
    settings.

#### Q39 How do you decide whether your linear regression model fits the data?
    Many solutions, such as use a loss function and check it situation, or use test data to verify 
    our model

~~#### Q40 What is the difference between squared error and absolute error?~~

#### Q41 What is Machine Learning?

    The simplest way to answer this question is – we give the data and equation to the machine. Ask the
    machine to look at the data and identify the coefficient values in an equation.

    For example for the linear regression y=mx+c, we give the data for the variable x, y and the machine
    learns about the values of m and c from the data.
    
#### Q42 How are confidence intervals constructed and how will you interpret them?
    Confidence interval is: under a certain confidence, the length of the area where the overall parameter
    is located. 

#### Q43 How will you explain logistic regression to an economist, physican scientist and biologist?

#### Q44 How can you overcome Overfitting?
    Regularization: add a regularizer or a penalty term.
    Cross Validation: Simple cross validation; S-folder cross validation; Leave-one-out cross validation.  

#### Q45 Differentiate between wide and tall data formats?
    Wide: data formats have lots of columns.
    Tall: data formats have lots of examples.

#### Q46 Is Naïve Bayes bad? If yes, under what aspects.

#### Q47 How would you develop a model to identify plagiarism?

#### Q48 How will you define the number of clusters in a clustering algorithm?

    Though the Clustering Algorithm is not specified, this question will mostly be asked in reference to
    K-Means clustering where “K” defines the number of clusters. The objective of clustering is to group 
    similar entities in a way that the entities within a group are similar to each other but the groups 
    are different from each other.

    For example, the following image shows three different groups.
    
![](https://s3.amazonaws.com/files.dezyre.com/images/blog/100+Data+Science+Interview+Questions+and+Answers+(General)/Data+Science+Interview+Questions+K-Means+Clustering.jpg)

    K-Mean Clustering Machine Learning Algorithm

    Within Sum of squares is generally used to explain the homogeneity within a cluster. If you plot WSS 
    for a range of number of clusters, you will get the plot shown below. The Graph is generally known as 
    Elbow Curve.
    
![](https://s3.amazonaws.com/files.dezyre.com/images/blog/100+Data+Science+Interview+Questions+and+Answers+(General)/Data+Science+Interview+Questions+K-Means.png)

    Red circled point in above graph i.e. Number of Cluster =6 is the point after which you don’t see any 
    decrement in WSS. This point is known as bending point and taken as K in K – Means.

    This is the widely used approach but few data scientists also use Hierarchical clustering first to 
    create dendograms and identify the distinct groups from there.
#### Q49 Is it better to have too many false negatives or too many false positives?
    It depends on the situation, for example, if we use the model for cancer detection, FN(False Negative)
    is more serious than FP(False Positive) because a FN could be verified in futher check, but
    FP maybe will let a patient be missed and delay the best treatment period.

#### Q50 Is it possible to perform logistic regression with Microsoft Excel?
    Yep, i must say Microsoft Excel is more and more powerful, and many data science could be 
    realized in simple way.

#### Q51 What do you understand by Fuzzy merging ? Which language will you use to handle it?

#### Q51 What is the difference between skewed and uniform distribution?

#### G52 You created a predictive model of a quantitative outcome variable using multiple regressions. What are the steps you would follow to validate the model?

    Since the question asked, is about post model building exercise, we will assume that you have 
    already tested for null hypothesis, multi collinearity and Standard error of coefficients.
    
    Once you have built the model, you should check for following –
- Global F-test to see the significance of group of independent variables on dependent variable
- R^2
- Adjusted R^2
- RMSE, MAPE

In addition to above mentioned quantitative metrics you should also check for-
- Residual plot
- Assumptions of linear regression 

#### Q54 What do you understand by Hypothesis in the content of Machine Learning?

#### Q55 What do you understand by Recall and Precision?

#### Q56 How will you find the right K for K-means?
    No any other way just do experiment on instance dataset, see the result of different K, find
    the better one. 

#### Q57 Why L1 regularizations causes parameter sparsity whereas L2 regularization does not?

    Regularizations in statistics or in the field of machine learning is used to include some extra 
    information in order to solve a problem in a better way. L1 & L2 regularizations are generally used 
    to add constraints to optimization problems.

![](https://s3.amazonaws.com/files.dezyre.com/images/blog/100+Data+Science+Interview+Questions+and+Answers+(General)/L1+L2+Regularizations.png)

    In the example shown above H0 is a hypothesis. If you observe, in L1 there is a high likelihood to 
    hit the corners as solutions while in L2, it doesn’t. So in L1 variables are penalized more as compared
    to L2 which results into sparsity.
    In other words, errors are squared in L2, so model sees higher error and tries to minimize that squared 
    error.
    
#### Q58 How can you deal with different types of seasonality in time series modelling?

#### Q59 In experimental design, is it necessary to do randomization? If yes, why?
    Normally yes, but never do it for time series dataset.

#### Q60 What do you understand by conjugate-prior with respect to Naïve Bayes?

#### Q61 Can you cite some examples where a false positive is important than a false negative?

    Before we start, let us understand what are false positives and what are false negatives.
    False Positives are the cases where you wrongly classified a non-event as an event a.k.a Type I error.
    And, False Negatives are the cases where you wrongly classify events as non-events, a.k.a Type II error.
    
![](https://s3.amazonaws.com/files.dezyre.com/images/blog/100+Data+Science+Interview+Questions+and+Answers+(General)/False+Positive+False+Negative.png)
    
    In medical field, assume you have to give chemo therapy to patients. Your lab tests patients for certain 
    vital information and based on those results they decide to give radiation therapy to a patient.
    Assume a patient comes to that hospital and he is tested positive for cancer (But he doesn’t have cancer) 
    based on lab prediction. What will happen to him? (Assuming Sensitivity is 1)

    One more example might come from marketing. Let’s say an ecommerce company decided to give $1000 Gift 
    voucher to the customers whom they assume to purchase at least $5000 worth of items. They send free voucher 
    mail directly to 100 customers without any minimum purchase condition because they assume to make at 
    least 20% profit on sold items above 5K.

    Now what if they have sent it to false positive cases? 
    
#### Q62 Can you cite some examples where a false negative important than a false positive?

    Assume there is an airport ‘A’ which has received high security threats and based on certain 
    characteristics they identify whether a particular passenger can be a threat or not. Due to shortage 
    of staff they decided to scan passenger being predicted as risk positives by their predictive model.
    What will happen if a true threat customer is being flagged as non-threat by airport model?
    
    Another example can be judicial system. What if Jury or judge decide to make a criminal go free?
    
    What if you rejected to marry a very good person based on your predictive model and you happen to
    meet him/her after few years and realize that you had a false negative?
    
#### Q63 Can you cite some examples where both false positive and false negatives are equally important?

    In the banking industry giving loans is the primary source of making money but at the same time if 
    your repayment rate is not good you will not make any profit, rather you will risk huge losses.
    
    Banks don’t want to lose good customers and at the same point of time they don’t want to acquire 
    bad customers. In this scenario both the false positives and false negatives become very important 
    to measure.

#### Q64 Can you explain the difference between a Test Set and a Validation Set?

    Validation set can be considered as a part of the training set as it is used for parameter selection
    and to avoid Overfitting of the model being built. On the other hand, test set is used for testing 
    or evaluating the performance of a trained machine leaning model.

    In simple terms ,the differences can be summarized as-
    
-   Training Set is to fit the parameters i.e. weights.
-   Test Set is to assess the performance of the model i.e. evaluating the predictive power and generalization.
-   Validation set is to tune the parameters.

#### Q65 What makes a dataset gold standard?
    

#### Q66 What do you understand by statistical power of sensitivity and how do you calculate it?

    Sensitivity is commonly used to validate the accuracy of a classifier (Logistic, SVM, RF etc.). 
    Sensitivity is nothing but “Predicted TRUE events/ Total events”. True events here are the events
    which were true and model also predicted them as true.
    
    Calculation of seasonality is pretty straight forward-
    
    ***Seasonality = True Positives /Positives in Actual Dependent Variable***
    
    Where, True positives are Positive events which are correctly classified as Positives.
    
#### Q67 What is the importance of having a selection bias?

#### Q68 Give some situations where you will use an SVM over a RandomForest Machine Learning algorithm and vice-versa.

    SVM and Random Forest are both used in classification problems.
    
    a)      If you are sure that your data is outlier free and clean then go for SVM. It is the 
    opposite - if your data might contain outliers then Random forest would be the best choice
    b)      Generally, SVM consumes more computational power than Random Forest, so if you are constrained 
    with memory go for Random Forest machine learning algorithm.
    c)  Random Forest gives you a very good idea of variable importance in your data, so if you want to 
    have variable importance then choose Random Forest machine learning algorithm.
    d)      Random Forest machine learning algorithms are preferred for multiclass problems.
    e)     SVM is preferred in multi-dimensional problem set - like text classification
    but as a good data scientist, you should experiment with both of them and test for accuracy or rather 
    you can use ensemble of many Machine Learning techniques.

#### Q69 What do you understand by feature vectors?

~~#### Q70 How do data management procedures like missing data handling make selection bias worse?~~

#### Q71 What are the advantages and disadvantages of using regularization methods like Ridge Regression?

~~#### Q72 What do you understand by long and wide data formats?~~

#### Q73 What do you understand by outliers and inliers? What would you do if you find them in your dataset?

~~#### Q74 Write a program in Python which takes input as the diameter of a coin and weight of the coin and produces output as the money value of the coin.

#### Q75 What are the basic assumptions to be made for linear regression?

    Normality of error distribution, statistical independence of errors, linearity and additivity.

#### Q76 Can you write the formula to calculat R-square?

    R-Square can be calculated using the below formular -
    1 - (Residual Sum of Squares/ Total Sum of Squares)

#### Q77 What is the advantage of performing dimensionality reduction before fitting an SVM?

    Support Vector Machine Learning Algorithm performs better in the reduced space. It is beneficial to 
    perform dimensionality reduction before fitting an SVM if the number of features is large when 
    compared to the number of observations.

#### Q78 How will you assess the statistical significance of an insight whether it is a real insight or just by chance?

    Statistical importance of an insight can be accessed using Hypothesis Testing.

## Machine Learning Interview Questions: Algorithms/Theory

#### Q79 What’s the trade-off between bias and variance?
    
    Bias is error due to erroneous or overly simplistic assumptions in the learning algorithm 
    you’re using. This can lead to the model underfitting your data, making it hard for it to have 
    high predictive accuracy and for you to generalize your knowledge from the training set to the 
    test set.

    Variance is error due to too much complexity in the learning algorithm you’re using. This leads 
    to the algorithm being highly sensitive to high degrees of variation in your training data, which 
    can lead your model to overfit the data. You’ll be carrying too much noise from your training data 
    for your model to be very useful for your test data.

    The bias-variance decomposition essentially decomposes the learning error from any algorithm by 
    adding the bias, the variance and a bit of irreducible error due to noise in the underlying dataset. 
    Essentially, if you make the model more complex and add more variables, you’ll lose bias but gain 
    some variance — in order to get the optimally reduced amount of error, you’ll have to tradeoff bias 
    and variance. You don’t want either high bias or high variance in your model.

#### Q80 What is the difference between supervised and unsupervised machine learning?
    
    Supervised learning requires training labeled data. For example, in order to do classification 
    (a supervised learning task), you’ll need to first label the data you’ll use to train the model
    to classify data into your labeled groups. Unsupervised learning, in contrast, does not require
    labeling data explicitly.

#### Q81 How is KNN different from k-means clustering?

    K-Nearest Neighbors is a supervised classification algorithm, while k-means clustering is an
    unsupervised clustering algorithm. While the mechanisms may seem similar at first, what this
    really means is that in order for K-Nearest Neighbors to work, you need labeled data you want to 
    classify an unlabeled point into (thus the nearest neighbor part). K-means clustering requires only
    a set of unlabeled points and a threshold: the algorithm will take unlabeled points and gradually
    learn how to cluster them into groups by computing the mean of the distance between different points.
    
    The critical difference here is that KNN needs labeled points and is thus supervised learning, while
    k-means doesn’t — and is thus unsupervised learning.

#### Q82 Explain how a ROC curve works.
    
    The ROC curve is a graphical representation of the contrast between true positive rates and the 
    false positive rate at various thresholds. It’s often used as a proxy for the trade-off between
    the sensitivity of the model (true positives) vs the fall-out or the probability it will trigger 
    a false alarm (false positives).
    
![](https://lh3.googleusercontent.com/zUWYO4VwGpoyu9oygT12F3hgZ30GxVY7sg_ZF46INrNbDutd9mVz9GnYIYGw2r1ZcbPLQXF4HV-uNXvQcVrP7Sg2BDDqRkaY3RAApumdXgH2mQZ8OCSgqqsVl7UDVjqwVFq224Z_)
    
#### Q83 Define precision and recall.
    
    Recall is also known as the true positive rate: the amount of positives your model claims 
    compared to the actual number of positives there are throughout the data. Precision is also 
    known as the positive predictive value, and it is a measure of the amount of accurate positives
    your model claims compared to the number of positives it actually claims. It can be easier to think
    of recall and precision in the context of a case where you’ve predicted that there were 10 apples
    and 5 oranges in a case of 10 apples. You’d have perfect recall (there are actually 10 apples, and
    you predicted there would be 10) but 66.7% precision because out of the 15 events you predicted,
    only 10 (the apples) are correct.

#### Q84 What is Bayes’ Theorem? How is it useful in a machine learning context?
    
    Bayes’ Theorem gives you the posterior probability of an event given what is known as prior knowledge.
    
    Mathematically, it’s expressed as the true positive rate of a condition sample divided by the sum of 
    the false positive rate of the population and the true positive rate of a condition. Say you had a 
    60% chance of actually having the flu after a flu test, but out of people who had the flu, the test 
    will be false 50% of the time, and the overall population only has a 5% chance of having the flu. 
    Would you actually have a 60% chance of having the flu after having a positive test?
    
    Bayes’ Theorem says no. It says that you have a (.6 * 0.05) (True Positive Rate of a Condition 
    Sample) / (.6*0.05)(True Positive Rate of a Condition Sample) + (.5*0.95) (False Positive Rate of
    a Population)  = 0.0594 or 5.94% chance of getting a flu.

    Bayes’ Theorem is the basis behind a branch of machine learning that most notably includes the
    Naive Bayes classifier. That’s something important to consider when you’re faced with machine 
    learning interview questions.

#### Q85 Why is “Naive” Bayes naive?

    Despite its practical applications, especially in text mining, Naive Bayes is considered “Naive” 
    because it makes an assumption that is virtually impossible to see in real-life data: the 
    conditional probability is calculated as the pure product of the individual probabilities of 
    components. This implies the absolute independence of features — a condition probably never met 
    in real life.

    As a Quora commenter put it whimsically, a Naive Bayes classifier that figured out that you liked 
    pickles and ice cream would probably naively recommend you a pickle ice cream.

#### Q86 Explain the difference between L1 and L2 regularization.

    L2 regularization tends to spread error among all the terms, while L1 is more binary/sparse, with
    many variables either being assigned a 1 or 0 in weighting. L1 corresponds to setting a Laplacean
    prior on the terms, while L2 corresponds to a Gaussian prior.
    
![](https://lh6.googleusercontent.com/vXUSHKE11Qpolek11IPPP6Fs-iU1-LeWtf5EXVdrfOl97ytug_cME-vLF1t4BNvoAppxfRhx4dNzHoKkdl8dfGVix4jc2hhvrtDG_wyuByxpVfeFZQdMH-INzG6RSi_9jkJLERto)

#### Q87 What’s your favorite algorithm, and can you explain it to me in less than a minute?

    This type of question tests your understanding of how to communicate complex and technical nuances 
    with poise and the ability to summarize quickly and efficiently. Make sure you have a choice and 
    make sure you can explain different algorithms so simply and effectively that a five-year-old could
    grasp the basics!

#### Q88 What’s the difference between Type I and Type II error?

    Don’t think that this is a trick question! Many machine learning interview questions will be an 
    attempt to lob basic questions at you just to make sure you’re on top of your game and you’ve
    prepared all of your bases.

    Type I error is a false positive, while Type II error is a false negative. Briefly stated, Type I 
    error means claiming something has happened when it hasn’t, while Type II error means that you claim 
    nothing is happening when in fact something is.

    A clever way to think about this is to think of Type I error as telling a man he is pregnant, while
    Type II error means you tell a pregnant woman she isn’t carrying a baby.

#### Q89 What’s a Fourier transform?

    A Fourier transform is a generic method to decompose generic functions into a superposition of symmetric
    functions. Or as this more intuitive tutorial puts it, given a smoothie, it’s how we find the recipe. The 
    Fourier transform finds the set of cycle speeds, amplitudes and phases to match any time signal. A Fourier
    transform converts a signal from time to frequency domain — it’s a very common way to extract features from
    audio signals or other time series such as sensor data.

#### Q90 What’s the difference between probability and likelihood?

![](https://lh3.googleusercontent.com/Yz2xAzLEEjtk62o9zatSDZJ7yBwgw-a1GtSNfAjJ3tq3OY5UbnxYUpNOqAuuKAUj8kVZaraIsr87kX83ejzg2y8DW9goGJbZuPc1Be_2VmGEEsNZ5JMioUw6Xke-KvYzp-sVrLCL)

#### Q91 What is deep learning, and how does it contrast with other machine learning algorithms?

    Deep learning is a subset of machine learning that is concerned with neural networks: how to use
    backpropagation and certain principles from neuroscience to more accurately model large sets of
    unlabelled or semi-structured data. In that sense, deep learning represents an unsupervised learning
    algorithm that learns representations of data through the use of neural nets.

#### Q92 What’s the difference between a generative and discriminative model?

    A generative model will learn categories of data while a discriminative model will simply learn the 
    distinction between different categories of data. Discriminative models will generally outperform 
    generative models on classification tasks.

#### Q93 What cross-validation technique would you use on a time series dataset?

    Instead of using standard k-folds cross-validation, you have to pay attention to the fact that a 
    time series is not randomly distributed data — it is inherently ordered by chronological order. If a 
    pattern emerges in later time periods for example, your model may still pick up on it even if that 
    effect doesn’t hold in earlier years!

    You’ll want to do something like forward chaining where you’ll be able to model on past data then
    look at forward-facing data.

    fold 1 : training [1], test [2]
    fold 2 : training [1 2], test [3]
    fold 3 : training [1 2 3], test [4]
    fold 4 : training [1 2 3 4], test [5]
    fold 5 : training [1 2 3 4 5], test [6]
#### Q94 How is a decision tree pruned?

    Pruning is what happens in decision trees when branches that have weak predictive power are removed 
    in order to reduce the complexity of the model and increase the predictive accuracy of a decision 
    tree model. Pruning can happen bottom-up and top-down, with approaches such as reduced error pruning 
    and cost complexity pruning.

    Reduced error pruning is perhaps the simplest version: replace each node. If it doesn’t decrease 
    predictive accuracy, keep it pruned. While simple, this heuristic actually comes pretty close to an 
    approach that would optimize for maximum accuracy.

#### Q95 Which is more important to you– model accuracy, or model performance?

    This question tests your grasp of the nuances of machine learning model performance! Machine learning 
    interview questions often look towards the details. There are models with higher accuracy that can 
    perform worse in predictive power — how does that make sense?
    
    Well, it has everything to do with how model accuracy is only a subset of model performance, and at 
    that, a sometimes misleading one. For example, if you wanted to detect fraud in a massive dataset with
    a sample of millions, a more accurate model would most likely predict no fraud at all if only a vast 
    minority of cases were fraud. However, this would be useless for a predictive model — a model designed
    to find fraud that asserted there was no fraud at all! Questions like this help you demonstrate that 
    you understand model accuracy isn’t the be-all and end-all of model performance.

#### Q96 What’s the F1 score? How would you use it?

    The F1 score is a measure of a model’s performance. It is a weighted average of the precision and recall
    of a model, with results tending to 1 being the best, and those tending to 0 being the worst. You would
    use it in classification tests where true negatives don’t matter much.

#### Q97 How would you handle an imbalanced dataset?

    An imbalanced dataset is when you have, for example, a classification test and 90% of the data is in one
    class. That leads to problems: an accuracy of 90% can be skewed if you have no predictive power on the
    other category of data! Here are a few tactics to get over the hump:

    1- Collect more data to even the imbalances in the dataset.

    2- Resample the dataset to correct for imbalances.

    3- Try a different algorithm altogether on your dataset.

    What’s important here is that you have a keen sense for what damage an unbalanced dataset can cause, 
    and how to balance that.

#### Q98 When should you use classification over regression?

    Classification produces discrete values and dataset to strict categories, while regression gives you
    continuous results that allow you to better distinguish differences between individual points. You would
    use classification over regression if you wanted your results to reflect the belongingness of data points 
    in your dataset to certain explicit categories (ex: If you wanted to know whether a name was male or 
    female rather than just how correlated they were with male and female names.)

#### Q99 Name an example where ensemble techniques might be useful.

    Ensemble techniques use a combination of learning algorithms to optimize better predictive performance.
    They typically reduce overfitting in models and make the model more robust (unlikely to be influenced by 
    small changes in the training data). 

    You could list some examples of ensemble methods, from bagging to boosting to a “bucket of models” method
    and demonstrate how they could increase predictive power.

#### Q100 How do you ensure you’re not overfitting with a model?

    This is a simple restatement of a fundamental problem in machine learning: the possibility of 
    overfitting training data and carrying the noise of that data through to the test set, thereby
    providing inaccurate generalizations.

    There are three main methods to avoid overfitting:

    1- Keep the model simpler: reduce variance by taking into account fewer variables and parameters, 
    thereby removing some of the noise in the training data.

    2- Use cross-validation techniques such as k-folds cross-validation.

    3- Use regularization techniques such as LASSO that penalize certain model parameters if they’re 
    likely to cause overfitting.

#### Q101 What evaluation approaches would you work to gauge the effectiveness of a machine learning model?

    You would first split the dataset into training and test sets, or perhaps use cross-validation
    techniques to further segment the dataset into composite sets of training and test sets within 
    the data. You should then implement a choice selection of performance metrics: here is a fairly
    comprehensive list. You could use measures such as the F1 score, the accuracy, and the confusion 
    matrix. What’s important here is to demonstrate that you understand the nuances of how a model is
    measured and how to choose the right performance measures for the right situations.

#### Q102 How would you evaluate a logistic regression model?

    A subsection of the question above. You have to demonstrate an understanding of what the typical goals 
    of a logistic regression are (classification, prediction etc.) and bring up a few examples and use cases.

#### Q103 What’s the “kernel trick” and how is it useful?

    The Kernel trick involves kernel functions that can enable in higher-dimension spaces without explicitly 
    calculating the coordinates of points within that dimension: instead, kernel functions compute the inner 
    products between the images of all pairs of data in a feature space. This allows them the very useful 
    attribute of calculating the coordinates of higher dimensions while being computationally cheaper than 
    the explicit calculation of said coordinates. Many algorithms can be expressed in terms of inner products.
    Using the kernel trick enables us effectively run  algorithms in a high-dimensional space with lower-dimensional data.

## Machine Learning Interview Questions: Programming
These machine learning interview questions test your knowledge of programming principles you need to 
implement machine learning principles in practice. Machine learning interview questions tend to be technical
questions that test your logic and programming skills: this section focuses more on the latter.

~~#### Q104 How do you handle missing or corrupted data in a dataset?~~

#### Q105 Do you have experience with Spark or big data tools for machine learning?

    You’ll want to get familiar with the meaning of big data for different companies and the different
    tools they’ll want. Spark is the big data tool most in demand now, able to handle immense datasets
    with speed. Be honest if you don’t have experience with the tools demanded, but also take a look at
    job descriptions and see what tools pop up: you’ll want to invest in familiarizing yourself with them.

#### Q106 Pick an algorithm. Write the psuedo-code for a parallel implementation.

    This kind of question demonstrates your ability to think in parallelism and how you could handle 
    concurrency in programming implementations dealing with big data. Take a look at pseudocode frameworks
    such as Peril-L and visualization tools such as Web Sequence Diagrams to help you demonstrate your 
    ability to write code that reflects parallelism.

#### Q107 What are some differences between a linked list and an array?

    An array is an ordered collection of objects. A linked list is a series of objects with pointers that
    direct how to process them sequentially. An array assumes that every element has the same size, unlike 
    the linked list. A linked list can more easily grow organically: an array has to be pre-defined or 
    re-defined for organic growth. Shuffling a linked list involves changing which points direct where — 
    meanwhile, shuffling an array is more complex and takes more memory.

#### Q108 Describe a hash table.

    A hash table is a data structure that produces an associative array. A key is mapped to certain values 
    through the use of a hash function. They are often used for tasks such as database indexing.

#### Q109 Which data visualization libraries do you use? What are your thoughts on the best data visualization tools?

    What’s important here is to define your views on how to properly visualize data and your personal 
    preferences when it comes to tools. Popular tools include R’s ggplot, Python’s seaborn and matplotlib,
    and tools such as Plot.ly and Tableau.
    
![](https://lh3.googleusercontent.com/79d5jkZBgpZPQa61A4e9opgfX2-mrxWxfQyswec3YxBouNEvAu8wYxjCXNQl-nRdBVQeuco1h-LZbxVblgS9h6bYLi6peoqSd2N7VW7BSeBgpmclKng6IRYEf9QkTMRJKMyPxrCT)

## Machine Learning Interview Questions: Company/Industry Specific

These machine learning interview questions deal with how to implement your general machine learning knowledge 
to a specific company’s requirements. You’ll be asked to create case studies and extend your knowledge of the
company and industry you’re applying for with your machine learning skills.

#### Q110 How would you implement a recommendation system for our company’s users?

    A lot of machine learning interview questions of this type will involve implementation of machine learning
    models to a company’s problems. You’ll have to research the company and its industry in-depth, especially 
    the revenue drivers the company has, and the types of users the company takes on in the context of the 
    industry it’s in.

#### Q111 How can we use your machine learning skills to generate revenue?

    This is a tricky question. The ideal answer would demonstrate knowledge of what drives the business and 
    how your skills could relate. For example, if you were interviewing for music-streaming startup Spotify, 
    you could remark that your skills at developing a better recommendation model would increase user retention,
    which would then increase revenue in the long run.

    The startup metrics Slideshare linked above will help you understand exactly what performance indicators 
    are important for startups and tech companies as they think about revenue and growth.

#### Q112 What do you think of our current data process?

    This kind of question requires you to listen carefully and impart feedback in a manner that is constructive 
    and insightful. Your interviewer is trying to gauge if you’d be a valuable member of their team and whether
    you grasp the nuances of why certain things are set the way they are in the company’s data process based on
    company- or industry-specific conditions. They’re trying to see if you can be an intellectual peer. Act 
    accordingly.

## Machine Learning Interview Questions: General Machine Learning Interest

This series of machine learning interview questions attempts to gauge your passion and interest in machine learning.
The right answers will serve as a testament for your commitment to being a lifelong learner in machine learning.

#### Q113 What are the last machine learning papers you’ve read?

    Keeping up with the latest scientific literature on machine learning is a must if you want to demonstrate
    interest in a machine learning position. This overview of deep learning in Nature by the scions of deep 
    learning themselves (from Hinton to Bengio to LeCun) can be a good reference paper and an overview of what’s 
    happening in deep learning — and the kind of paper you might want to cite.

#### Q114 Do you have research experience in machine learning?

    Related to the last point, most organizations hiring for machine learning positions will look for your 
    formal experience in the field. Research papers, co-authored or supervised by leaders in the field, can make 
    the difference between you being hired and not. Make sure you have a summary of your research experience 
    and papers ready — and an explanation for your background and lack of formal research experience if you don’t.

#### Q115 What are your favorite use cases of machine learning models?

    The Quora thread above contains some examples, such as decision trees that categorize people into different 
    tiers of intelligence based on IQ scores. Make sure that you have a few examples in mind and describe what 
    resonated with you. It’s important that you demonstrate an interest in how machine learning is implemented.

#### Q116 How would you approach the “Netflix Prize” competition?

    The Netflix Prize was a famed competition where Netflix offered $1,000,000 for a better collaborative 
    filtering algorithm. The team that won called BellKor had a 10% improvement and used an ensemble of different
    methods to win. Some familiarity with the case and its solution will help demonstrate you’ve paid attention 
    to machine learning for a while.

#### Q117 Where do you usually source datasets?

    Machine learning interview questions like these try to get at the heart of your machine learning interest.
    Somebody who is truly passionate about machine learning will have gone off and done side projects on their own, 
    and have a good idea of what great datasets are out there. If you’re missing any, check out Quandl for economic
    and financial data, and Kaggle’s Datasets collection for another great list.

#### Q118 How do you think Google is training data for self-driving cars?

    Machine learning interview questions like this one really test your knowledge of different machine learning 
    methods, and your inventiveness if you don’t know the answer. Google is currently using recaptcha to source 
    labelled data on storefronts and traffic signs. They are also building on training data collected by Sebastian
    Thrun at GoogleX — some of which was obtained by his grad students driving buggies on desert dunes!

#### Q119 How would you simulate the approach AlphaGo took to beat Lee Sidol at Go?

    AlphaGo beating Lee Sidol, the best human player at Go, in a best-of-five series was a truly seminal event
    in the history of machine learning and deep learning. The Nature paper above describes how this was accomplished
    with “Monte-Carlo tree search with deep neural networks that have been trained by supervised learning,
    from human expert games, and by reinforcement learning from games of self-play.”


[Reference from dezyre](https://www.dezyre.com/article/100-data-science-interview-questions-and-answers-general-for-2017/184 "悬停显示")

[Rererence from Springbord](https://www.springboard.com/blog/machine-learning-interview-questions/?from=message&isappinstalled=0 "悬停显示")

Reference: Deep Learning (Ian Goodfellow, Yoshua Bengio and Aaron Courville) -- MIT




