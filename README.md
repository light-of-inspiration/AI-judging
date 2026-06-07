# AI-judging
Comparative analysis between human &amp; AI decision mechanism
  
## 1. Introduction
To determine whether AI can judge like human in judicial scenes, we conduct a comparative analysis.  
  
The methodology is to make use of machine learning models, and cognitive science theories to obtain the general decision model of human and AI in a complex judicial scene.  
  
## 2. Task explanations
We select the scene of distinguishing between Assisting in cybercrime activities (帮助信息网络犯罪活动罪, ‘the assistive crime’ in short) and Crime of concealing and disguising proceeds of crime (掩饰隐瞒犯罪所得犯罪所得收益罪, ‘the concealing crime’ in short). It is a very controversial topic in China’s judicial practice and theory, for both crimes involving similar description but resulting in huge differences in the final sentence application.  
  
> Paragraph 1, Article 287b Whoever, knowing that another person is committing or is going to commit a crime by using information networks, still provides the person with technical support such as internet access, server hosting, network storage, communications and transmission, or provides assistance in advertising or promotion, or payment and settlement, etc., if the circumstances are serious, shall be sentenced to fixed-term imprisonment of not more than 3 years or short-term custody, and concurrently, a fine, or shall be sentenced to a fine only.
>   
> Paragraph 1, Article 312 Whoever knowingly hides, transfers, purchases, sells for another person, or covers up or conceals by other means criminal gains and the proceeds derived therefrom shall be sentenced to fixed-term imprisonment of not more than 3 years, short-term custody, or non-custodial correction, and concurrently, a fine, or shall be sentenced to a fine only. If the circumstances are serious, the offender shall be sentenced to fixed-term imprisonment of not less than 3 years but not more than 7 years, and concurrently, a fine.
>   
> See [Criminal Law, PRC](http://en.npc.gov.cn.cdurl.cn/2020-12/26/c_921604_13.htm "(2020)")  

## 3. Dataset production
Here, we are NOT aiming to settle this problem, but to investigate how the judges and AI will act in this task.  

### 3.1 Structured data for conventional model training
Therefore, we collect the precedent data from the iCourt database, and search for all the cases in the past one year (2025) which include both the keywords ‘the crime of assisting information network criminal activities’ and ‘the crime of concealing or covering up proceeds of crime and profits derived from crime’, then exclude the repeated (71 records) or exceptional data (3 records) to get the cleaned data (951 records). For sampling method, considering the distribution of the last number of the precedent id is normally uniformly distributed, hence we can select the precedent with the end of ‘0’ as the sampled data (95 samples with 162 defendants’ records).  
  
By extracting common factors that affect the application of law and the main elements involved in theoretical disputes, we have produced `LAIC.xlsx` for machine learning models.  
  
> We attended the LAIC contest with the first version of the manuscript, while ‘LAIC.xlsx’ refers to the usage, but NOT the data resource. We make use of the data according to the Berne Convention and Article 24 of the Copyright Law, PRC.   
  
### 3.2 Unstructured data for analysis via natural language
Among all the precedents, we select one with 12 defendants, where 2 of them have similar senteence application results but different result of charge. Hence we take this case as the unstructured data for comparative analysis.  
  
> (1)**Defendant Tang**. Tang has offered 3 bank cards to others to receiving and transferring illicit proceeds, earning illegal profit of 8,500 yuan, which involves 1 telecommunication fraud transfer (30,000 yuan in total). After withdrawing cash and retaining a kickback of 1,200 yuan, he transferred the money to other accomplices. Tang voluntarily surrendered upon telephone summons and truthfully confessed to the crimes, constituting voluntary surrender that may warrant leniency or reduced punishment. His guilty plea and acceptance of penalties further qualify him for lenient treatment.
>   
> In this judgement, Tang was convicted of the concealing crime and sentenced to 8 months of fixed-term imprisonment with a 1-year probation period, along with a fine of 5,000 yuan.  
  
> (2)**Defendant Li**. Li operated a livestock farm and, being unable to secure a loan from traditional banking institutions, sought financial assistance through Huang. In compliance with the latter’s stipulations, Li established a corporate account and relinquished control by providing the associated U-Key, account details, and password. This account was subsequently utilized to receive transfers from the implicated accounts of Tang and others, specifically processing a single transaction of 610,000 yuan linked to telecommunication fraud. Li is classified as a recidivist. However, given that Li has confessed to the offence and accepted punishment in accordance with the law, a mitigated sentence may be considered.
>   
> In this judgement, Li was convicted of the assistive crime and sentenced to 8 months of fixed-term imprisonment, along with a fine of 5,000 yuan.
>
> See the precedent (2025) Qing 0105 No. 20 in the first instance criminal trials, ‘Zhang et al. prosecuted for the concealing crime’, released by People’s Court of Chengbei District, Xining City. 
〔参见（2025）青0105刑初20号判决书“张某某等掩饰、隐瞒犯罪所得案”西宁市城北区人民法院〕  
  
## 4. Others
We also upload some code, environment, running results for reference.  

In the folder `ChatData`, we have shown how the LLMs generation content for comparison is made. we give the `prompt.txt` and the `task.txt` to form the prompt, and transfer AI's answer into the table in the manuscript.  
   
In addition, as you can see, it is a Chinese dataset, and it is temporarily impossible to translate all of them, but online translators may help. If you have any other questions, contact us through: <u> fountainofideas@foxmail.com </u> .  
  
---  
Updated June, 2026  












