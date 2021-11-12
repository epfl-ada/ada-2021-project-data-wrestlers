# Donald Trump’s favorite topics

## Abstract
The idea is to look at quotes from Donald Trump from during the 2016 presidential campaign and his time in office. We would then extract topics from these quotes, such as economics, ecology, etc. We would look at the percentage of speech allocated to each extracted topic and the most frequent words, and how it changes over time. The goal would be to identify events in the news which provoke changes in Trump’s speech (such as Covid). We would also see if there is a big difference in his favorite topics between before and after his election. Perhaps certain topics are talked about a lot during the campaign but are ignores after the election, or the opposite. We would then interpret this analysis by seeing how accurately speech reflects his political agenda.

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

## Methods
1. Split Trump's quotes into small time chunks (about a week, to be determined) and apply Latent Dirichlet Allocation (LDA), considering the quotes contained in one time chunk as a single document. Then compare the time chunks before and after the election, based on how frequent each topic is, what words are the most relevant in each topic, etc.
2. Interpretation: we look at the topics contained in Trump's political agenda and compare them to the topics extracted using LDA in question 1.
3. We use the same analysis as in question 1, but now looking at how the mixture of topics varies over time, and interpreting the fluctuations using the list of key events during Trump's presidency.
4. Apply a pre-trained sentiment analysis model on quotations about Trump to identify positive and negative opinions. We look over time to compare with the topics that Trump addresses over time. We also see if different population groups emerge and identify these groups with the metadata we have on speakers. We also compare the population impact results with the polls on Trump's approval ratings. 
5. Extract newspaper names from the URL. We see if Trump's quotes are more or less cited in certain newspapers. We also look at whether the topics of Trump's quotes are distributed uniformly or not between the newspapers. Finally, we look at the quotes about Trump to see if certain magazines are more or less in favor of Trump. 
6. We take the two groups (quotes from Trump and quotes about Trump) and extract their topics using LDA. Then on each topic and on each group we apply the pre-trained sentiment analysis model and we compare between the two groups.

## Proposed timeline

## Organization within the team

## Questions for TAs
