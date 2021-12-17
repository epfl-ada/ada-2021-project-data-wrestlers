# What image of Trump emerges from news articles between 2015 and 2020?

## Abstract

Our project focuses on the analysis of Donald Trump's presidency through the quotes in the Quotebank dataset. The idea is to analyze quotes from and about Donald Trump during the 2016 presidential campaign and his time in office. We will use statistical models to extract topics from these quotes, such as economy, ecology, etc. and study their evolution over time, how they change, how they differ in the words of Trump and in the quotes about him. One goal is to identify events in the news which could be correlated with changes in Trump’s speech (such as Covid). We also plan to analyze the difference between his campaign and his mandate as well as how much his speech reflects his initial political agenda over time by observing topics rarefying and others emerging. We will also study how people who talk about Trump feel about him.

## Research Questions
- **Trump's speech:**
1. What were the topics Donald Trump talked about most during the presidential campaign (starting in 2015)? and during his time in office (20/01/2017 - 20/01/2021)? Can we observe a change in these topics between before and after he was elected?
2. Can we note different periods? Can we correlate changes in the ideas expressed by Trump with external events?  

- **External view of Trump:**
3. Can we quantify Trump's popularity over time and correlate it with external events? Can we see the impact of his speech on different groups in the population?
4. In which newspapers are Trump's quotations published? Is there a link between the newspapers that published the most quotations of Trump and the political affiliation of these newspapers?


## Additional datasets
- [List of key events during Trump's presidency](https://millercenter.org/president/trump/key-events): used to correlate with changes in topics
- [Political agenda during campaign](https://ballotpedia.org/Donald_Trump_presidential_campaign,_2020): used to compare with topics discussed during campaign
- [Trump approval ratings](https://projects.fivethirtyeight.com/trump-approval-ratings/): used to compare with sentiment of speech about Trump
- [Speaker metadata](https://drive.google.com/drive/folders/1VAFHacZFh0oxSxilgNByb1nlNsqznUf0): used to compare how different groups (ethnic, political, etc.) in the population talk about Trump

The first three additional datasets will not be extracted but only used as comparison points.

## Methods

The first step was to extract from the Quotebank dataset the quotes which could be about or from Trump. As described in part 1.1 of `notebook_milestone3.ipynb`, we read all the provided files (`Quotebank/quotes-20*.json.bz2`) and selected the samples containing "Trump" as a potential author (see `data/quotes-from-trump.json.bz2`) and the ones mentioning "trump" in the quotation lowered quotation text (see `data/quotes-about-trump.json.bz2`). Additional text preprocessing and filtering based on author classification probabilities was applied in part 1 and 2 of `notebook_milestone3.ipynb`.

The next steps to investigate our research questions are the following:

1. Split Trump's quotes into time chunks of 100 days and apply Empath, considering the quotes contained in one time chunk as a single document. Then compare the time chunks before and after the election, based on how frequent each topic is, what words are the most relevant in each topic, etc.
2. We use the question 1 analysis, and we look at how the topics varies over time, and interpreting the fluctuations using the list of key events during Trump's presidency.
3. Apply two pre-trained sentiment analysis models (Vader and Flair, to compare the result obtained in both) on quotations about Trump to identify positive and negative opinions. We look over time to compare with the topics mentioned in Trump's speech. We also see if different population groups emerge and identify these groups with the metadata we have on speakers. We also compare the population impact results with the polls on Trump's approval ratings. 
4. Extract newspaper names from the URL. We see if Trump's quotes are more or less cited in certain newspapers. We also look at the quotes about Trump to see if certain magazines are more or less in favor of Trump by using the same sentiment analysis as before. 


Note that we already implemented the function to apply LDA on variable sized time chunks of quotes, sadly the parameters are too complicated to set, so we used Empath instead.
While inquiring about pre-trained sentiment analysis models, two of them seem quite good for us: Vader and Flair. Both can take into account negation, intensifiers and can predict a sentiment that it has never seen before. Moreover, their uses seems accessible to us. The most striking difference between them is that Flair returns only "positive" or "negative" for a sentence, while Vader return a compound between -1 and 1, which is more accurate.

## Timeline

**15.11-21.11**: Machine Learning data generation

- Apply, fine-tune and collect results of LDA models (questions 1 and 4)
- Apply and collect results of pre-trained sentiment analysis models (questions 3 and 4)

**22.11-28.11**: Questions 1, and 2

- Analyze results and discuss research questions

**29.11-5.12**: Question 3

- Analyze results and discuss research questions

**6.12-12.12**: Final analysis and datastory

- Finalize the analysis for all research questions
- Formalize the datastory
- First website design discussion

**12.12-17.12**: Website implementation

- Build the github pages website

**17.12**: Milestone 3 submission

## Organization within the team:

- *Yazid Makmani*: Website, speaker metadata handling, data story
- *Félicie Giraud-Sauveur*: Sentiment analysis, data story
- *Eliot Walt*: Newspaper analysis, data story
- *Thomas Berkane*: Topic evolution, data story
