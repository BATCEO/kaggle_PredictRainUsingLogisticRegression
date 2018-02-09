Kaggle上面的一个例子

# kattle_Predict_rain_using_Logistic_Regression
使用逻辑回归 预测西雅图的降雨情况

[阅读原文](https://www.kaggle.com/anudeepbommireddy/predict-rain-using-logistic-regression/code)

除了咖啡、垃圾食品和科技公司，西雅图最出名的就是经常下雨。这个数据集包含了从1948年1月1日到2017年12月12日的每日降雨模式的完整记录。  
数据集信息  
DATE:日期  
PRCP:降水量，单位为英寸  
TMAX:最高温度    
TMIN:最低温度   
RAIN:是否下雨  

#### 读取数据
```r
sw <- read.csv("C:\\Users\\pactera\\Desktop\\seattleWeather_1948-2017.csv")
```
#### 检查数据维度 查看一共多少行 多少列
```r
dim(sw)
```
#### 移除RAIN为空的列
```r
sw1 <- sw[-c(which(is.na(sw$RAIN))),]
dim(sw1)
```
#### 构建回归模型  RAIN随TMAX+TMIN变化而变化
#glm被用于拟合广义线性模型，通过给出线性预测器的符号描述和对误差分布的描述来指定。  
#glm(formula, family = gaussian, data, weights, subset,  
#na.action, start = NULL, etastart, mustart, offset,  
#control = list(...), model = TRUE, method = "glm.fit",  
#x = FALSE, y = TRUE, contrasts = NULL, ...)  
```r
lrmodel <- glm(RAIN~TMAX+TMIN, family = binomial, data=sw1)
summary(lrmodel)
```
#**************************************************************  
#family 选不同值 结果也不同    
#lrmodel <- glm(RAIN~TMAX+TMIN,family = binomial ,data=sw1)  
#summary(lrmodel)  
#  
#Call:  
#glm(formula = RAIN ~ TMAX + TMIN, family = binomial, data = sw1)  
#  
#Deviance Residuals:   
#Min       1Q   Median       3Q      Max    
#-2.4818  -0.8052  -0.2547   0.8406   3.3328    
#  
#Coefficients:  
#Estimate Std. Error z value Pr(>|z|)        
#(Intercept)  2.872580   0.082985   34.62   <2e-16 ***   
#TMAX        -0.253155   0.003470  -72.95   <2e-16 ***  
#TMIN         0.261419   0.004248   61.54   <2e-16 ***  
#---  
#Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1  
#  
#(Dispersion parameter for binomial family taken to be 1)  
#
#Null deviance: 34865  on 25547  degrees of freedom  
#Residual deviance: 25443  on 25545  degrees of freedom  
#AIC: 25449  
#  
#Number of Fisher Scoring iterations: 5  
#
#**************************************************************  
#lrmodel <- glm(RAIN~TMAX+TMIN,family = gaussian ,data=sw1)  
#summary(lrmodel)  
#  
#Call:  
#glm(formula = RAIN ~ TMAX + TMIN, family = gaussian, data = sw1)  
#  
#Deviance Residuals:     
#Min        1Q    Median        3Q       Max    
#-1.42613  -0.36540   0.02495   0.35426   1.36078    
#  
#Coefficients:  
#Estimate Std. Error t value Pr(>|t|)      
#(Intercept)  0.9543122  0.0133531   71.47   <2e-16 ***  
#TMAX        -0.0401538  0.0003986 -100.74   <2e-16 ***  
#TMIN         0.0418574  0.0005725   73.11   <2e-16 ***  
#---  
#Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1  
#  
#(Dispersion parameter for gaussian family taken to be 0.1716587)  
#  
#Null deviance: 6249.5  on 25547  degrees of freedom  
#Residual deviance: 4385.0  on 25545  degrees of freedom  
#AIC: 27485  
#  
#Number of Fisher Scoring iterations: 2   
#**************************************************************  

```r
#install.packages("caret")  
library(caret)  
#cat 相当于print  
cat("Variable_Importance:")  
#计算变量的重要性  
varImp(lrmodel)  

#type 选择预测之后的输出结果，这个参数能用在binomial数据，也就是响应变量是二分型的时候，  
# 这个参数选成type＝response，表示输出结果预测响应变量为1的概率。就是下雨的概率  
predicted_values <- predict(lrmodel, type="response")  
cat("Predicted_Values:\n")  
predicted_values[1:10]  

#阈值  
threshold=0.5  
cat("Threshold_Value:")  
threshold  

#预测结果概率大于0.5定为会下雨  小于0.5定为不会下雨  
predicted_class <- ifelse(predict(lrmodel, type="response")>threshold,TRUE,FALSE)  
cat("Predicted_Class:\n")  
predicted_class[1:10]  

#实际降雨情况  

actual_values <- lrmodel$y  
cat("Actual_Values:\n")
actual_values[1:10]

conf_matrix <- table(predicted_class,actual_values)
cat("Confusion_Matrix:\n")
conf_matrix
```
#预测与实际直接的比对  
#> conf_matrix
#               actual_values
#predicted_class     0     1
#          FALSE 11424  3140
#          TRUE   3224  7760
#> 


#accuracy准确性

accuracy <- (conf_matrix[1,1]+conf_matrix[2,2])/(sum(conf_matrix)) #正确预测结果/总的预测结果
cat("Accuracy_of_the_Model:")
accuracy

#灵敏度
sensi <- (conf_matrix[1,1])/(conf_matrix[1,1]+conf_matrix[1,2])
cat("Sensitivity:")
sensi

#特异性
speci <- (conf_matrix[2,2])/(conf_matrix[2,1]+conf_matrix[2,2])
cat("Specificity:")
speci

#install.packages("pROC")
library(pROC)
#roc 这是pROC包的主要功能。 它建立一个ROC曲线并返回一个“roc”对象，一个“roc”类的列表。 这个对象可以打印，绘图或传递给函数auc，ci，smooth.roc和coords。 此外，可以将两个roc对象与roc.test进行比较
roccurve <- roc(actual_values, predicted_values)
plot(roccurve)
<img src="img/a.png"></img>  
#auc 该功能用梯形法则计算ROC曲线下面积（AUC）的数值。 有两种语法是可能的：roc函数中的类“roc”的一个对象，或者两个向量（响应，预测器）或公式（响应预测器）。 默认情况下，计算总AUC，但部分ROC曲线可以用partial.auc指定。
auc(roccurve)
```
