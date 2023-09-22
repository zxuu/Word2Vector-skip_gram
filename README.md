# Word2Vector-skip_gram
默认用skip-gram训练自己的词向量。用CBOW请将cg设置为0
>model = word2vec.Word2Vec(sentences, size=50, sg=1, window=10, min_count=5, workers=4, iter=5)
### 数据
一行一个数据。如下：
```bash
当希望工程救助的百万儿童成长起来，科教兴国蔚然成风时，今天有收藏价值的书你没买，明日就叫你悔不当初！
藏书本来就是所有传统收藏门类中的第一大户，只是我们结束温饱的时间太短而已。
因有关日寇在京掠夺文物详情，藏界较为重视，也是我们收藏北京史料中的要件之一。
```
### 训练后的词向量文件
会生成corpusSegDone.vector训练好的词向量文件
```bash
领导 -0.29350913 0.3383447 0.6914202 -0.2709373 0.3843902 0.87859905 0.49312145 ...
学习 -0.27468246 0.54872274 0.4444796 -0.75887495 0.97562027 0.5284329 0.30799964 ...
比赛 0.37641197 1.21803 -0.44004828 0.069927156 0.9466083 0.35451618 -0.053610377 ...
教育 -0.014285841 0.6079104 0.21085194 -0.69194657 0.33619738 0.38108802 0.14272486 ...
````
### 依赖包
<<<<<<< HEAD
>pip install gensim==3.7.1    # 推荐这个版本否则可能报错
>pip install jieba

参考这篇[博客](https://blog.csdn.net/qq_42491242/article/details/104782989 "https://blog.csdn.net/qq_42491242/article/details/104782989")
还有一篇博客是用训练的[使用GloVe生成中文词向量](https://blog.csdn.net/weixin_40952784/article/details/100729036 'https://blog.csdn.net/weixin_40952784/article/details/100729036')
=======
```bash
pip install gensim==3.7.1    # 推荐这个版本否则可能报错
pip install jieba
```
参考这篇[博客](https://blog.csdn.net/qq_42491242/article/details/104782989 "https://blog.csdn.net/qq_42491242/article/details/104782989")
>>>>>>> f2dcf181e4f27c191759b8ea83fb8df7569fd038
