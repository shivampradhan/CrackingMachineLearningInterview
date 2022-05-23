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
