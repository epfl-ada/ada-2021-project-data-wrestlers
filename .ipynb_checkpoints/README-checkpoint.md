# Donald Trump’s favorite topics

## Abstract

Our project focuses on the analysis of Donald Trump presidency through the quotes in the Quotebank dataset. The idea is to analyze quotes from and about Donald Trump during the 2016 presidential campaign and his time in office. We will use statistical models to extract topics from these quotes, such as economy, ecology, etc. and study their evolution over time, how they change, how they differ in the words of Trump and in the quotes about him. One goal is to identify events in the news which could be correlated with changes in Trump’s speech (such as Covid). We also plan to analyze the difference between his campaign and his mandate as well as how much his speech reflects his initial political agenda over time by observing topics rarefying and others emerging.

## Research Questions
- **Trump's speech:**
1. What were the topics Donald Trump talked about most during the presidential campaign (starting in 2015)? and during his time in office (20/01/2017 - 20/01/2021)? Can we observe a change in these topics between before and after he was elected?
2. How well aligned is Trump's speech with his political agenda/program? How close was Donald Trump's presidency to his initial program?
3. Can we note different periods? Can we correlate changes in the ideas expressed by Trump with external events?  

- **External view of Trump:**
4. Can we quantify Trump's popularity over time and correlate it with external events? Can we see the impact of his speech on different groups in the population?
5. In which newspapers are Trump's quotations published? Is there a link between the newspapers that published the most quotations of Trump and the political affiliation of these newspapers?
6. What insight can we get from the difference between quotations from Trump and quotations about him?

## Proposed additional datasets
- [List of key events during Trump's presidency](https://millercenter.org/president/trump/key-events): used to correlate with changes in topics
- [Political agenda during campaign](https://ballotpedia.org/Donald_Trump_presidential_campaign,_2020): used to compare with topics discussed during campaign
- [Trump approval ratings](https://projects.fivethirtyeight.com/trump-approval-ratings/): used to compare with sentiment of speech about Trump
- [Speaker metadata](https://drive.google.com/drive/folders/1VAFHacZFh0oxSxilgNByb1nlNsqznUf0): used to compare how different groups (ethnic, political, etc.) in the population talk about Trump
The first three additional datasets will not be extracted but only used as comparison points.

## Methods

The first step was to extract from the Quotebank dataset the quotes which could be about or from Trump. As described in part 1.1 of `notebook_milestone2_Trump_analysis.ipynb`, we read all the provided files (`Quotebank/quotes-20*.json.bz2`) and selected the samples containing "Trump" as a potential author (see `data/quotes-from-trump.json.bz2`) and the ones mentioning "trump" in the quotation lowered quotation text (see `data/quotes-about-trump.json.bz2`). Additional text preprocessing and filtering based on author classification probabilities was applied in part 1 and 2 of `notebook_milestone2_Trump_analysis.ipynb`.

The next steps that we are planning to take to investigate our research questions are the following:

1. Split Trump's quotes into small time chunks (about a week, to be determined) and apply Latent Dirichlet Allocation (LDA), considering the quotes contained in one time chunk as a single document. Then compare the time chunks before and after the election, based on how frequent each topic is, what words are the most relevant in each topic, etc.
2. Interpretation: we look at the topics contained in Trump's political agenda and compare them to the topics extracted using LDA in question 1.
3. We use the same analysis as in question 1, but now looking at how the mixture of topics varies over time, and interpreting the fluctuations using the list of key events during Trump's presidency.
4. Apply a pre-trained sentiment analysis model on quotations about Trump to identify positive and negative opinions. We look over time to compare with the topics mentioned in Trump's speech. We also see if different population groups emerge and identify these groups with the metadata we have on speakers. We also compare the population impact results with the polls on Trump's approval ratings. 
5. Extract newspaper names from the URL. We see if Trump's quotes are more or less cited in certain newspapers. We also look at whether the topics of Trump's quotes are distributed uniformly or not between the newspapers. Finally, we look at the quotes about Trump to see if certain magazines are more or less in favor of Trump. 
6. We take the two groups (quotes from Trump and quotes about Trump) and extract their topics using LDA. Then, on each topic and on each group we apply the pre-trained sentiment analysis model and we compare between the two groups.

Note that we already implemented the function to apply LDA on variable sized time chunks of quotes. It can be found at the end of  `notebook_milestone2_Trump_analysis.ipynb ` under the name `timechunk_lda()`.
We started to inquire about the pre-trained sentiment analysis model and we thought about using Flair because it can take into account negation, intensifiers and can predict a sentiment that it has never seen before. Moreover its use seems accessible to us.

## Proposed timeline

**15.11-21.11**: Machine Learning data generation

- Apply, fine-tune and collect results of LDA models (questions 1 and 6)
- Apply and collect results of pre-trained sentiment analysis models (questions 4 and 6)

**22.11-28.11**: Questions 1, 2 and 3

- Analyze results and discuss research questions

**29.11-5.12**: Questions 4, 5 and 6

- Analyze results and discuss research questions

**6.12-12.12**: Final analysis and datastory

- Finalize the analysis for all research questions
- Formalize the datastory
- First website design discussion

**12.12-17.12**: Website implementation

- Build the github pages website

**17.12**: Milestone 3 submission

## Organization within the team

The remaining work can be divided into the following internal milestones:

- Machine Learning data generation (15.11-21.11)
- Research questions analysis and discussion (22.11-5.12)
- Datastory creation (6.12-12.12)
- Website design and implementation (13.12-17.12)

## Questions for TAs

At this point we have no additional question to the TAs.