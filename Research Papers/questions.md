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
