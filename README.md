# News Sentiment Analysis Predictor

## Project Overview
Stock prices constantly fluctuate due to changes in consumer supply and demand. If a consumer is in the market for a stock, the market price will trend upwards. On the other hand, if consumers are trying to sell a stock, the market price will fall. The impact of news media on stock prices has been a crucial factor in investors' decision-making process. News articles with a negative sentiment to them can cause people to sell their positions and vice versa, positive articles can persuade those to buy up more positions. These variations in news sentiment are imperative in an investor's verdict on whether or not to change their position in a stock.

The question that I sought to answer and discover is how I could analyze news articles and determine if the overall sentiment of the article is positive or negative and how that could influence investors on whether or not to purchase that stock. I wanted to determine if I could parse through news articles, with already predetermined positive/negative sentiment scores and create a confidence table on whether an investor should buy or sell a company's stock based on that sentiment. The machine learning technique used in this project was CatBoost/XGBoost. With these confidence scores, I hope to have a better understanding of how news articles can directly impact a company's stock price action. 

## Repository
The repository for my project can be found at
+ https://github.com/RayyanWaseem1/News-Sentiment-Analysis-Predictor

## Dataset
Two datasets were used in this project. The Market Data dataset contained financial indicators such as opening and closing prices, volume, and returns for each company. The News Data dataset contained timestamped news events from 2007 with indicators such as headlines, sentiment scores, and relevance metrics. 

For the preprocessing of my data, I converted each timestamp into the datetime format and merged the two datasets based on that feature. The features selected for this model were extracted from both datasets. From the Market Data dataset, I selected 'volume', 'close', 'open', 'returnsClosePrevRaw1', 'returnsOpenPrevRaw1', and 'returnsOpenNextMktres10'. From the News Data dataset, I selected 'sentimentNegative', 'sentimentNeutral', 'sentimentPositive', 'relevance', and 'firstMenstionSentence'. 

## Machine Learning Methodology
The methodology used for this project was CatBoost/XGBoost. I sought to find a confidence value for each of the 100 companies. The context in which this project was conducted was to cater to the financial/portfolio management industry. The reason that I chose to use a boosting ensemble method, particularly CatBoost, is because of its ability to handle the categorical features within the two datasets that I used. Because my data consisted of both qualitative and categorical features, CatBoost further eliminated the need for any sort of extensive preprocessing (e.g., one-hot encoding). As well as this, for any missing data in my datasets, CatBoost was able to inherently manage them without the need for any imputation. I also preferred the use of CatBoost because of my smaller dataset with only 100 samples, which could have been susceptible to overfitting with other methods. 

After using CatBoost, I used XGBoost because it is traditionally well-suited for financial data. Also, XGBoost was used to strongly rank which features were deemed the most important in my model. As well as this, XGBoost was used as a benchmark against CatBoost to assess my model's performance within feature engineering. 

The main libraries used were CatBoost, XGBoost, NumPy, Pandas, Matplotlib, and Sklearn. 

## Results
Initially, I was curious as to how a negative, neutral, or positive sentiment could influence the price action of a company's stock. In order to answer this question, I started by merging both datasets. This allowed me to improve feature engineering and prediction accuracy by combining both categorical and numerical market indicators. 

After combining both datasets, I selected the most relevant features from the merged dataset for training and testing my model. The features that I had deemed relevant were:

['volume', 'close', 'open', 'returnsClosePrevRaw1', 'returnsOpenPrevRaw1', 'returnsClosePrevMktres1', 'returnsClosePrevRaw10', 'returnsOpenPrevRaw10', 'returnsClosePrevMktres10', 'returnsOpenPrevMktres10', 'MA_15MA', 'MA_30MA', 'MA_45MA', 'MA_60MA', 'MA_80MA']

The reason 'volume' was selected as a feature to be trained on was due to its ability to measure market activity through momentum shifts. A high trading volume could indicate a strong interest and signal a good opportunity to enter the market. Whereas a drop in volume could suggest a diminished interest, leading an investor to sell. To pair with 'volume', I decided to select 'close' and 'open' just as a way to capture the price of each company's stock within the time frame. Finally, 'returnsClosePrevRaw1' and 'returnsOpenPrevRaw1' were also chosen as short-term return metrics, which were crucial for the prediction of immediate movements. This was also applied for 'returnsClosePrevRaw10' and 'returnsOpenPrevRaw10' as longer-term return metrics. 

My data was then split into training and testing and applied to CatBoost. My training data was scored at 1.0, whereas my testing data was reported to be 0.725. While my testing accuracy was not as high as I would have wanted it to be, and could be improved, I was pleased to see that there was no overfitting despite the small-sized dataset that I used. 

After testing my model through CatBoost, I then used XGBoost to strongly rank the importance of the features that I used. To my surprise 'open' was by far the most important of all features, subsequently followed by 'returnsOpenPrevRaw10' and 'returnsOpenPrevRaw1', respectively.

<img width="530" alt="Screenshot 2025-02-22 at 6 26 29 PM" src="https://github.com/user-attachments/assets/f149f059-39bd-4d1a-be56-b068236b5f03" />

Finally, I predicted the probabilities for each class amongst my testing data through CatBoost. This can be seen through 'pred_1 = model.predict_proba(X_test)'. Each prediction was added to the variable 'bagged_prediction' as a means to accumulate the values from each ensemble model. The variable 'bagged_prediction' was then averaged across the four different models to reduce the impact of any outliers from any single particular model. I was then able to use the value of 'bagged_prediction' and create a final confidence value for each company in my dataset. The confidence values ranged from -1, strong confidence in a price drop, to 1, strong confidence in a price increase. 

<img width="669" alt="Screenshot 2025-02-22 at 6 36 27 PM" src="https://github.com/user-attachments/assets/30cb159a-7a03-4ae5-b826-fc20bcd961ff" />

The confidence values can be used as part of a trading strategy to evaluate a risk threshold. Depending on the risk appetite of an investor, they can use these values as a means of buying or selling. This can lead to a better justification of stronger trading decisions and suggest caution on any decisions that might increase risk.

<img width="488" alt="Screenshot 2025-02-22 at 6 38 54 PM" src="https://github.com/user-attachments/assets/bba00286-61e3-4fc0-9366-1797b1afaf81" />

## Lessons Learned
Throughout the duration of this project, I learned a lot regarding ensemble methods, particularly CatBoost and XGBoost. When I first began exploring how to solve my question on news sentiment analysis, I did not have a strong understanding of just how important ensemble methods could be, especially for categorical data. After a lot of trial and error, I am fairly pleased with the results that I have conducted. 

While working on this project, most of the challenges that I faced were determining which features I would use to conduct the sentiment analysis. As well as this, from the beginning, I feared that my dataset would be too small and I would keep running into my model overfitting. At first, I planned on just using XGBoost as the primary method of solving this problem. However, after learning about the efficiency of CatBoost on categorical data and how CatBoost could handle smaller datasets without the need for imputation, I decided to make it my primary method. 

In conclusion, it can be seen that news sentiment analysis can have a direct impact on the price action of a company's stock. Whether the sentiment was good, bad, or neutral, a confidence value was calculated for investors to use as a means to enhance their trading strategy and risk appetite. 

To those reading and reviewing my first project, I hope to have provided a solid foundational understanding of ensemble methods like CatBoost and XGBoost and how they both can be used in following any company's stock movement based on news analysis. 
