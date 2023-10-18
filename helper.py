import sys
import numpy as np
import seaborn as sb
from pca import pca
from math import sqrt
from tabulate import tabulate
from matplotlib import pyplot as plt

from pandas import DataFrame, MultiIndex, concat, DatetimeIndex, Series

from scipy import stats #확률 분포, 통계 분석, 신호 처리, 최적화, 선형 대수 등의 다양한 기능을 포함
from scipy.stats import t, pearsonr, spearmanr
from scipy.stats import shapiro, normaltest, ks_2samp, bartlett, fligner, levene, chi2_contingency

from statsmodels.formula.api import ols, logit
from statsmodels.tsa.stattools import adfuller  # adfuller를 사용해 데이터가 정상성을 만족하는지 판단 가능
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, accuracy_score, recall_score, precision_score, f1_score, r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split    #train/test 데이터 분리
from sklearn.linear_model import LinearRegression   #선형회귀
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.model_selection import GridSearchCV
#--------------------------------------------------
# 모듈 불러오기
# import sys
# import os
# sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
# from helper import normality_test, equal_variance_test, independence_test, all_test
#--------------------------------------------------


# 시각화 폰트 - tabulate의 스타일을 지정해 변경할 수 있다(tablefmt="psql")
def prettyPrint(df, headers="keys", tablefmt="psql", numalign="right", title="value"):
    # print(tabulate(df, headers=headers, tablefmt=tablefmt, numalign=numalign))
    if isinstance(df, Series):
        df = DataFrame(df, columns=[title])
    print(tabulate(df, headers=headers, tablefmt=tablefmt, numalign=numalign))


# 결측치 정제(평균 값)
def replaceMissingValue(df, strategy='mean'):
    """
    결측치 정제

    Parameters
    -------
    - df: 데이터 프레임
    - strategy: 결측치 대체 전략(mean, median, most_frequent). 기본값은 mean

    Returns
    -------
    - re_df: 정제된 데이터 프레임
    """
    imr = SimpleImputer(missing_values=np.nan, strategy=strategy)
    df_imr = imr.fit_transform(df.values)
    re_df = DataFrame(df_imr, index=df.index, columns=df.columns)
    return re_df


# 분산분석을 수행하기 위한 정규성 검정
def normality_test(*any):
    """
    분산분석을 수행하기 위한 정규성을 검정 한다.
    Parameters
    -------
    - any: 필드들
    Returns
    -------
    - df: 검정 결과 데이터 프레임
    """
    names = []

    result = {
        'statistic': [],
        'p-value': [],
        'result': []
    }
    for i in any:
        s, p = shapiro(i)
        result['statistic'].append(s)
        result['p-value'].append(p)
        result['result'].append(p > 0.05)
        names.append(('정규성', 'shapiro', i.name))

    for i in any:
        s, p = normaltest(i)
        result['statistic'].append(s)
        result['p-value'].append(p)
        result['result'].append(p > 0.05)
        names.append(('정규성', 'normaltest', i.name))

    n = len(any)

    for i in range(0, n):
        j = i + 1 if i < n - 1 else 0

        s, p = ks_2samp(any[i], any[j])
        result['statistic'].append(s)
        result['p-value'].append(p)
        result['result'].append(p > 0.05)
        names.append(('정규성', 'ks_2samp', f'{any[i].name} vs {any[j].name}'))

    return DataFrame(result, index=MultiIndex.from_tuples(names, names=['condition', 'test', 'field']))


# 분산분석을 수행하기 위한 등분산성 검정
def equal_variance_test(*any):
    """
    분산분석을 수행하기 위한 등분산성을 검정 한다.
    Parameters
    -------
    - any: 필드들
    Returns
    -------
    - df: 검정 결과 데이터 프레임
    """
    # statistic=1.333315753388535, pvalue=0.2633161881599037
    s1, p1 = bartlett(*any)
    s2, p2 = fligner(*any)
    s3, p3 = levene(*any)
    names = []
    for i in any:
        names.append(i.name)

    fix = " vs "
    name = fix.join(names)
    index = [['등분산성', 'Bartlett', name], ['등분산성', 'Fligner', name], ['등분산성', 'Levene', name]]

    df = DataFrame({
        'statistic': [s1, s2, s3],
        'p-value': [p1, p2, p3],
        'result': [p1 > 0.05, p2 > 0.05, p3 > 0.05]
    }, index=MultiIndex.from_tuples(index, names=['condition', 'test', 'field']))

    return df


# 분산분석을 수행하기 위한 독립성 검정
def independence_test(*any):
    """
    분산분석을 수행하기 위한 독립성을 검정한다.
    Parameters
    -------
    - any: 필드들
    Returns
    -------
    - df: 검정 결과 데이터 프레임
    """
    df = DataFrame(any).T
    result = chi2_contingency(df)
    names = []
    for i in any:
        names.append(i.name)
    fix = " vs "
    name = fix.join(names)

    index = [['독립성', 'Chi2', name]]

    df = DataFrame({
        'statistic': [result.statistic],
        'p-value': [result.pvalue],
        'result': [result.pvalue > 0.05]
    }, index=MultiIndex.from_tuples(index, names=['condition', 'test', 'field']))

    return df


# 정규성, 등분산성, 독립성을 모두 검정
def all_test(*any):
    """
    정규성, 등분산성, 독립성을 모두 검정한다.
    Parameters
    -------
    - any: 필드들
    Returns
    -------
    - df: 검정 결과 데이터 프레임
    """
    return concat([normality_test(*any), equal_variance_test(*any), independence_test(*any)])


# IQR(Interquartile Range)를 이용한 이상치 경계값 계산
def getIq(field, isPrint=True):
    """
    IQR(Interquartile Range)를 이용한 이상치 경계값 계산

    Parameters
    ------- 
    - field: 데이터 프레임의 필드

    Returns
    -------
    - 결측치경계: 이상치 경계값 리스트
    """
    q1 = field.quantile(q=0.25)
    q3 = field.quantile(q=0.75)
    iqr = q3 - q1
    하한 = q1 - 1.5 * iqr
    상한 = q3 + 1.5 * iqr
    극단치경계 = [하한, 상한]   # 시계열을 위해 추가

    
    df = DataFrame({
        "극단치 경계": [하한, 상한]
    }, index=['하한', '상한'])

    if isPrint:
        prettyPrint(df)
    else:
        return 극단치경계   # 시계열을 위해 추가 여기까지


# 이상치를 판별하여 결측치로 치환
def replaceOutlier(df, fieldName):
    """
    이상치를 판별하여 결측치로 치환

    Parameters
    -------
    - df: 데이터 프레임
    - fieldName: 이상치를 판별할 필드명

    Returns
    -------
    - cdf : 결측치를 이상치로 치환한 데이터 프레임
    """
    cdf = df.copy()

    # fieldName이 List가 아니면 List로 변환
    if not isinstance(fieldName, list):
        fieldName = [fieldName]

    for f in fieldName:
        결측치경계 = getIq(cdf[f])
        cdf.loc[cdf[f] < 결측치경계[0], f] = np.nan
        cdf.loc[cdf[f] > 결측치경계[1], f] = np.nan

    return cdf


# 데이터 프레임에서 지정된 필드를 범주형으로 변경
def setCategory(df, fields=[], labelling=True):
    """
    데이터 프레임에서 지정된 필드를 범주형으로 변경한다.

    Parameters
    -------
    - df: 데이터 프레임
    - fields: 범주형으로 변경할 필드명 리스트. 기본값은 빈 리스트(전체 필드 대상)

    Returns
    -------
    - cdf: 범주형으로 변경된 데이터 프레임
    """
    cdf = df.copy()
    # 데이터 프레임의 변수명을 리스트로 변환
    ilist = list(cdf.dtypes.index)
    # 데이터 프레임의 변수형을 리스트로 변환
    vlist = list(cdf.dtypes.values)

    # 변수형에 대한 반복 처리
    for i, v in enumerate(vlist):
        # 변수형이 object이면?
        if v == 'object':
            # 변수명을 가져온다.
            field_name = ilist[i]

            # 대상 필드 목록이 설정되지 않거나(전체필드 대상), 현재 필드가 대상 필드목록에 포함되어 있지 않다면?
            if not fields or field_name not in fields:
                continue

            # 가져온 변수명에 대해 값의 종류별로 빈도를 카운트 한 후 인덱스 이름순으로 정렬
            # vc = cdf[field_name].value_counts().sort_index()
            # print(vc)

            # 인덱스 이름순으로 정렬된 값의 종류별로 반복 처리
            # for ii, vv in enumerate(list(vc.index)):
            #     # 일련번호값으로 치환
            #     cdf.loc[cdf[field_name] == vv, field_name] = ii

            # 해당 변수의 데이터 타입을 범주형으로 변환
            cdf[field_name] = cdf[field_name].astype('category')

            if labelling:
                mydict = {}

                for i, v in enumerate(cdf[field_name].dtypes.categories):
                    mydict[v] = i
                
                cdf[field_name] = cdf[field_name].map(mydict).astype(int)

    return cdf


# 불용어를 제거
def clearStopwords(nouns, stopwords_file_path="wordcloud/stopwords-ko.txt"):
    """
    불용어를 제거한다.

    Parameters
    -------
    - nouns: 명사 리스트
    - stopwords_file_path: 불용어 파일 경로. 기본값은 wordcloud/stopwords-ko.txt

    Returns
    -------
    - data_set: 불용어가 제거된 명사 리스트
    """
    with open(stopwords_file_path, 'r', encoding='utf-8') as f:
        stopwords = f.readlines()

        for i, v in enumerate(stopwords):
            stopwords[i] = v.strip()

    data_set = []

    for v in nouns:
        if v not in stopwords:
            data_set.append(v)

    return data_set


# 신뢰구간 생성(시계열을 위해 변경)
def getConfidenceInterval(data, clevel=0.95, isPrint=True):
    """
    신뢰구간 계산

    Parameters
    -------
    - data: 데이터
    - clevel: 신뢰수준. 기본값은 0.95

    Returns
    -------
    - cmin: 신뢰구간 하한
    - cmax: 신뢰구간 상한
    """
    n = len(data)                           # 샘플 사이즈
    dof = n - 1                             # 자유도
    sample_mean = data.mean()               # 표본 평균
    sample_std = data.std(ddof=1)           # 표본 표준 편차
    sample_std_error = sample_std / sqrt(n)  # 표본 표준오차

    # 신뢰구간
    cmin, cmax = t.interval(
        clevel, dof, loc=sample_mean, scale=sample_std_error)

    if isPrint:
        df = DataFrame({
            "신뢰구간": [cmin, cmax]
        }, index=['하한', '상한'])

        prettyPrint(df)
    else:
        return (cmin, cmax)


# 분산분석을 수행하기 위한 정규성을 검정
def normalityTest(*any, isPrint=True):
    """
    분산분석을 수행하기 위한 정규성을 검정 한다.

    Parameters
    -------
    - any: 필드들

    Returns
    -------
    - df: 검정 결과 데이터 프레임
    """
    names = []

    result = {
        'field': [],
        'test': [],
        'statistic': [],
        'p-value': [],
        'result': []
    }
    for i in any:
        s, p = shapiro(i)
        result['field'].append(i.name)
        result['test'].append('shapiro')
        result['statistic'].append(s)
        result['p-value'].append(p)
        result['result'].append(p > 0.05)
        names.append('정규성')

    for i in any:
        s, p = normaltest(i)
        result['field'].append(i.name)
        result['test'].append('shapiro')
        result['statistic'].append(s)
        result['p-value'].append(p)
        result['result'].append(p > 0.05)
        names.append('정규성')

    n = len(any)

    for i in range(0, n):
        j = i + 1 if i < n - 1 else 0

        s, p = ks_2samp(any[i], any[j])
        result['field'].append(f'{any[i].name} vs {any[j].name}')
        result['test'].append('ks_2samp')
        result['statistic'].append(s)
        result['p-value'].append(p)
        result['result'].append(p > 0.05)
        names.append('정규성')

    rdf = DataFrame(result, index=names)

    if isPrint:
        prettyPrint(rdf)
    else:
        return rdf


# 분산분석을 수행하기 위한 등분산성을 검
def equalVarianceTest(*any, isPrint=True):
    """
    분산분석을 수행하기 위한 등분산성을 검정 한다.

    Parameters
    -------
    - any: 필드들

    Returns
    -------
    - df: 검정 결과 데이터 프레임
    """
    s1, p1 = bartlett(*any)
    s2, p2 = fligner(*any)
    s3, p3 = levene(*any)

    names = []

    for i in any:
        names.append(i.name)

    fix = " vs "
    name = fix.join(names)
    index = ['등분산성', '등분산성', '등분산성']

    df = DataFrame({
        'field': [name, name, name],
        'test': ['Bartlett', 'Fligner', 'Levene'],
        'statistic': [s1, s2, s3],
        'p-value': [p1, p2, p3],
        'result': [p1 > 0.05, p2 > 0.05, p3 > 0.05]
    }, index=index)

    if isPrint:
        prettyPrint(df)
    else:
        return df


# 분산분석을 수행하기 위한 독립성을 검정
def independenceTest(*any, isPrint=True):
    """
    분산분석을 수행하기 위한 독립성을 검정한다.

    Parameters
    -------
    - any: 필드들

    Returns
    -------
    - df: 검정 결과 데이터 프레임
    """
    df = DataFrame(any).T
    result = chi2_contingency(df)

    names = []

    for i in any:
        names.append(i.name)

    fix = " vs "
    name = fix.join(names)

    index = ['독립성']

    df = DataFrame({
        'field': [name],
        'test': ['Chi2'],
        'statistic': [result.statistic],
        'p-value': [result.pvalue],
        'result': [result.pvalue > 0.05]
    }, index=index)

    if isPrint:
        prettyPrint(df)
    else:
        return df


# 정규성, 등분산성, 독립성을 모두 검정
def allTest(*any, isPrint=True):
    """
    정규성, 등분산성, 독립성을 모두 검정한다.

    Parameters
    -------
    - any: 필드들

    Returns
    -------
    - df: 검정 결과 데이터 프레임
    """
    cc = concat([normalityTest(*any, isPrint=False), equalVarianceTest(*any, isPrint=False), independenceTest(*any, isPrint=False)])

    if isPrint:
        prettyPrint(cc)
    else:
        return cc


#------------------------------
# 피어슨 상관분석
#------------------------------
# 피어슨 상관계수를 사용하여 상관분석을 수행
def pearson_r(df, isPrint=True):
    """
    피어슨 상관계수를 사용하여 상관분석을 수행한다.

    Parameters
    -------
    - df: 데이터 프레임

    Returns
    -------
    - rdf: 상관분석 결과 데이터 프레임
    """
    names = df.columns
    n = len(names)
    pv = 0.05

    data = []

    for i in range(0, n):
        # 기본적으로 i 다음 위치를 의미하지만 i가 마지막 인덱스일 경우 0으로 설정
        j = i + 1 if i < n - 1 else 0

        fields = names[i] + ' vs ' + names[j]
        s, p = pearsonr(df[names[i]], df[names[j]])
        result = p < pv

        data.append({'fields': fields, 'statistic': s,
                    'pvalue': p, 'result': result})

    rdf = DataFrame(data)
    rdf.set_index('fields', inplace=True)

    if isPrint:
        prettyPrint(rdf)
    else:
        return rdf
#------------------------------
# 스피어만 상관분석
#------------------------------
# 스피어만 상관계수를 사용하여 상관분석을 수행
def spearman_r(df, isPrint=True):
    """
    스피어만 상관계수를 사용하여 상관분석을 수행한다.

    Parameters
    -------
    - df: 데이터 프레임

    Returns
    -------
    - rdf: 상관분석 결과 데이터 프레임
    """
    names = df.columns
    n = len(names)
    pv = 0.05

    data = []

    for i in range(0, n):
        # 기본적으로 i 다음 위치를 의미하지만 i가 마지막 인덱스일 경우 0으로 설정
        j = i + 1 if i < n - 1 else 0

        fields = names[i] + ' vs ' + names[j]
        s, p = spearmanr(df[names[i]], df[names[j]])
        result = p < pv

        data.append({'fields': fields, 'statistic': s,
                    'pvalue': p, 'result': result})

    rdf = DataFrame(data)
    rdf.set_index('fields', inplace=True)

    if isPrint:
        prettyPrint(rdf)
    else:
        return rdf


'''
# 모듈 지정
설명력
평균절대오차
평균제곱오차
평균오차
평균 절대 백분 오차 비율
평균 비율 오차
'''
class RegMetric:
    def __init__(self, y, y_pred):
        # 설명력
        self._r2 = r2_score(y, y_pred)
        # 평균절대오차
        self._mae = mean_absolute_error(y, y_pred)
        # 평균 제곱 오차
        self._mse = mean_squared_error(y, y_pred)
        # 평균 오차
        self._rmse = np.sqrt(self._mse)

        # 평균 절대 백분오차 비율
        if type(y) == Series:
            self._mape = np.mean(np.abs((y.values - y_pred) / y.values) * 100)
        else:
            self._mape = np.mean(np.abs((y - y_pred) / y) * 100)

        # 평균 비율 오차
        if type(y) == Series:   
            self._mpe = np.mean((y.values - y_pred) / y.values * 100)
        else:
            self._mpe = np.mean((y - y_pred) / y * 100)

    @property
    def r2(self):
        return self._r2

    @r2.setter
    def r2(self, value):
        self._r2 = value

    @property
    def mae(self):
        return self._mae

    @mae.setter
    def mae(self, value):
        self._mae = value

    @property
    def mse(self):
        return self._mse

    @mse.setter
    def mse(self, value):
        self._mse = value

    @property
    def rmse(self):
        return self._rmse

    @rmse.setter
    def rmse(self, value):
        self._rmse = value

    @property
    def mape(self):
        return self._mape

    @mape.setter
    def mape(self, value):
        self._mape = value

    @property
    def mpe(self):
        return self._mpe

    @mpe.setter
    def mpe(self, value):
        self._mpe = value     
        

# 회귀분석 결과를 위한 class
class OlsResult:
    def __init__(self):
        self._x_train = None    #머신러닝 회귀 갱신
        self._y_train = None
        self._train_pred = None
        self._x_test = None
        self._y_test = None
        self._test_pred = None  #머신러닝 회귀 갱신

        self._model = None
        self._fit = None
        self._summary = None
        self._table = None
        self._result = None
        self._goodness = None
        self._varstr = None
        self._coef = None   #기울기
        self._intercept = None  #절편
        self._trainRegMetric = None #학습
        self._testRegMetric = None  #테스트

    #머신러닝 회귀 갱신
    @property
    def x_train(self):
        return self._x_train

    @x_train.setter
    def x_train(self, value):
        self._x_train = value

    @property
    def y_train(self):
        return self._y_train

    @y_train.setter
    def y_train(self, value):
        self._y_train = value

    @property
    def train_pred(self):
        return self._train_pred

    @train_pred.setter
    def train_pred(self, value):
        self._train_pred = value

    @property
    def x_test(self):
        return self._x_test

    @x_test.setter
    def x_test(self, value):
        self._x_test = value

    @property
    def y_test(self):
        return self._y_test

    @y_test.setter
    def y_test(self, value):
        self._y_test = value

    @property
    def test_pred(self):
        return self._test_pred

    @test_pred.setter
    def test_pred(self, value):
        self._test_pred = value
    #머신러닝 회귀 갱신

    @property
    def model(self):
        """
        분석모델
        """
        return self._model

    @model.setter
    def model(self, value):
        self._model = value

    @property
    def fit(self):
        """
        분석결과 객체
        """
        return self._fit

    @fit.setter
    def fit(self, value):
        self._fit = value

    @property
    def summary(self):
        """
        분석결과 요약 보고
        """
        return self._summary

    @summary.setter
    def summary(self, value):
        self._summary = value

    @property
    def table(self):
        """
        결과표
        """
        return self._table

    @table.setter
    def table(self, value):
        self._table = value

    @property
    def result(self):
        """
        결과표 부가 설명
        """
        return self._result

    @result.setter
    def result(self, value):
        self._result = value

    @property
    def goodness(self):
        """
        모형 적합도 보고
        """
        return self._goodness

    @goodness.setter
    def goodness(self, value):
        self._goodness = value

    @property
    def varstr(self):
        """
        독립변수 보고
        """
        return self._varstr

    @varstr.setter
    def varstr(self, value):
        self._varstr = value

    @property
    def coef(self):   #기울기
        return self._coef

    @coef.setter
    def coef(self, value):
        self._coef = value

    @property
    def intercept(self):    #절편
        return self._intercept

    @intercept.setter
    def intercept(self, value):
        self._intercept = value

    @property
    def trainRegMetric(self):   #학습
        return self._trainRegMetric

    @trainRegMetric.setter
    def trainRegMetric(self, value):
        self._trainRegMetric = value

    @property
    def testRegMetric(self):    #테스트
        return self._testRegMetric

    @testRegMetric.setter
    def testRegMetric(self, value):
        self._testRegMetric = value

    def setRegMetric(self, y_train, y_train_pred, y_test=None, y_test_pred=None):
        self.trainRegMetric = RegMetric(y_train, y_train_pred)
        
        if y_test is not None and y_test_pred is not None:
            self.testRegMetric = RegMetric(y_test, y_test_pred)


# 회귀분석을 수행
def myOls(data, y=None, x=None, expr=None):
    """
    회귀분석을 수행한다.

    Parameters
    -------
    - data : 데이터 프레임
    - y: 종속변수 이름
    - x: 독립변수의 이름들(리스트)
    """

    # DF 복사
    df = data.copy()

    # 종속변수~독립변수1+독립변수2+독립변수3+... 형태의 식을 생성
    if not expr:
        # 독립변수의 이름이 리스트가 아니라면 리스트로 변환
        if type(x) != list:
            x = [x]
        expr = "%s~%s" % (y, "+".join(x))
    else:
        x = []
        p = expr.find('~')
        y = expr[:p].strip()
        x_tmp = expr[p+1:]
        x_list = x_tmp.split('+')

        for i in x_list:
            k = i.strip()

            if k:
                x.append(k)


    # 회귀모델 생성
    model = ols(expr, data=data)
    # 분석 수행
    fit = model.fit()

    # 파이썬 분석결과를 변수에 저장한다.
    summary = fit.summary()

    # 첫 번째, 세 번째 표의 내용을 딕셔너리로 분해
    my = {}

    for k in range(0, 3, 2):
        items = summary.tables[k].data
        # print(items)

        for item in items:
            # print(item)
            n = len(item)

            for i in range(0, n, 2):
                key = item[i].strip()[:-1]
                value = item[i+1].strip()

                if key and value:
                    my[key] = value

    # 두 번째 표의 내용을 딕셔너리로 분해하여 my에 추가
    my['variables'] = []
    name_list = list(data.columns)
    #print(name_list)

    for i, v in enumerate(summary.tables[1].data):
        if i == 0:
            continue

        # 변수의 이름
        name = v[0].strip()

        vif = 0

        # Intercept는 제외
        if name in name_list:
            # 변수의 이름 목록에서 현재 변수가 몇 번째 항목인지 찾기 
            j = name_list.index(name)
            vif = variance_inflation_factor(data, j)

        my['variables'].append({
            "name": name,
            "coef": v[1].strip(),
            "std err": v[2].strip(),
            "t": v[3].strip(),
            "P-value": v[4].strip(),
            "Beta": 0,
            "VIF": vif,
        })

    # 결과표를 데이터프레임으로 구성
    mylist = []
    yname_list = []
    xname_list = []

    for i in my['variables']:
        if i['name'] == 'Intercept':
            continue

        yname_list.append(y)
        xname_list.append(i['name'])

        item = {
            "B": i['coef'],
            "표준오차": i['std err'],
            "β": i['Beta'],
            "t": "%s*" % i['t'],
            "유의확률": i['P-value'],
            "VIF": i["VIF"]
        }

        mylist.append(item)

    table = DataFrame(mylist,
                   index=MultiIndex.from_arrays([yname_list, xname_list], names=['종속변수', '독립변수']))
    
    # 분석결과
    result = "𝑅(%s), 𝑅^2(%s), 𝐹(%s), 유의확률(%s), Durbin-Watson(%s)" % (my['R-squared'], my['Adj. R-squared'], my['F-statistic'], my['Prob (F-statistic)'], my['Durbin-Watson'])

    # 모형 적합도 보고
    goodness = "%s에 대하여 %s로 예측하는 회귀분석을 실시한 결과, 이 회귀모형은 통계적으로 %s(F(%s,%s) = %s, p < 0.05)." % (y, ",".join(x), "유의하다" if float(my['Prob (F-statistic)']) < 0.05 else "유의하지 않다", my['Df Model'], my['Df Residuals'], my['F-statistic'])

    # 독립변수 보고
    varstr = []

    for i, v in enumerate(my['variables']):
        if i == 0:
            continue
        
        s = "%s의 회귀계수는 %s(p%s0.05)로, %s에 대하여 %s."
        k = s % (v['name'], v['coef'], "<" if float(v['P-value']) < 0.05 else '>', y, '유의미한 예측변인인 것으로 나타났다' if float(v['P-value']) < 0.05 else '유의하지 않은 예측변인인 것으로 나타났다')

        varstr.append(k)

    ols_result = OlsResult()
    ols_result.model = model
    ols_result.fit = fit
    ols_result.summary = summary
    ols_result.table = table
    ols_result.result = result
    ols_result.goodness = goodness
    ols_result.varstr = varstr

    return ols_result


# 데이터 프레임을 표준화-정규화(scaling)
def scalling(df, yname=None):
    """
    데이터 프레임을 표준화 한다.

    Parameters
    -------
    - df: 데이터 프레임
    - yname: 종속변수 이름

    Returns
    -------
    - x_train_std_df: 표준화된 독립변수 데이터 프레임
    - y_train_std_df: 표준화된 종속변수 데이터 프레임
    """
    # 평소에는 yname을 제거한 항목을 사용
    # yname이 있지 않다면 df를 복사
    x_train = df.drop([yname], axis=1) if yname else df.copy()
    x_train_std = StandardScaler().fit_transform(x_train)
    x_train_std_df = DataFrame(x_train_std, columns=x_train.columns)
    
    if yname:
        y_train = df.filter([yname])
        y_train_std = StandardScaler().fit_transform(y_train)
        y_train_std_df = DataFrame(y_train_std, columns=y_train.columns)

    if yname:
        result = (x_train_std_df, y_train_std_df)
    else:
        result = x_train_std_df

    return result


#------------------------------
# 시계열데이터분석
# 회귀분석에 필요한 요인 선정을 위해
# 주성분 분석을 수행
#------------------------------
# 주성분 분석-PCA 분석
def getBestFeatures(x_train_std_df):
    pca_model = pca()
    fit = pca_model.fit_transform(x_train_std_df)
    topfeat_df = fit['topfeat']
    
    best = topfeat_df.query("type=='best'")
    feature = list(set(list(best['feature'])))
    
    return (feature, topfeat_df)


# 로지스틱 회귀분석 결과를 위한 class
class LogitResult:
    def __init__(self):
        self._model = None    
        self._fit = None
        self._summary = None
        self._prs = None
        self._cmdf = None
        self._result_df = None
        self._odds_rate_df = None

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        self._model = value

    @property
    def fit(self):
        return self._fit

    @fit.setter
    def fit(self, value):
        self._fit = value

    @property
    def summary(self):
        return self._summary

    @summary.setter
    def summary(self, value):
        self._summary = value

    @property
    def prs(self):
        return self._prs

    @prs.setter
    def prs(self, value):
        self._prs = value

    @property
    def cmdf(self):
        return self._cmdf

    @cmdf.setter
    def cmdf(self, value):
        self._cmdf = value

    @property
    def result_df(self):
        return self._result_df

    @result_df.setter
    def result_df(self, value):
        self._result_df = value

    @property
    def odds_rate_df(self):
        return self._odds_rate_df

    @odds_rate_df.setter
    def odds_rate_df(self, value):
        self._odds_rate_df = value

# 로지스틱 회귀분석을 수행
def myLogit(data, y, x, subset=None):
    """
    로지스틱 회귀분석을 수행한다.

    Parameters
    -------
    - data : 데이터 프레임
    - y: 종속변수 이름
    - x: 독립변수의 이름들(리스트)
    """

    # 데이터프레임 복사
    df = data.copy()

    # 독립변수의 이름이 리스트가 아니라면 리스트로 변환
    if type(x) != list:
        x = [x]

    # 종속변수~독립변수1+독립변수2+독립변수3+... 형태의 식을 생성
    expr = "%s~%s" % (y, "+".join(x))

    # 회귀모델 생성
    model = logit(expr, data=df)
    # 분석 수행
    fit = model.fit()

    # 파이썬 분석결과를 변수에 저장한다.
    summary = fit.summary()

    # 의사결정계수
    prs = fit.prsquared

    # 예측결과를 데이터프레임에 추가
    df['예측값'] = fit.predict(df.drop([y], axis=1))
    df['예측결과'] = df['예측값'] > 0.5

    # 혼동행렬
    cm = confusion_matrix(df[y], df['예측결과'])
    tn, fp, fn, tp = cm.ravel()
    cmdf = DataFrame([[tn, fn], [fp, tp]], index=['True', 'False'], columns=['Negative', 'Positive'])

    # RAS
    ras = roc_auc_score(df[y], df['예측결과'])

    # 위양성율, 재현율, 임계값(사용안함)
    fpr, tpr, thresholds = roc_curve(df[y], df['예측결과'])

    # 정확도
    acc = accuracy_score(df[y], df['예측결과'])

    # 정밀도
    pre = precision_score(df[y], df['예측결과'])

    # 재현율
    recall = recall_score(df[y], df['예측결과'])

    # F1 score
    f1 = f1_score(df[y], df['예측결과'])

    # 위양성율
    fallout = fp / (fp + tn)

    # 특이성
    spe = 1 - fallout

    result_df = DataFrame({'설명력(Pseudo-Rsqe)': [fit.prsquared], '정확도(Accuracy)':[acc], '정밀도(Precision)':[pre], '재현율(Recall, TPR)':[recall], '위양성율(Fallout, FPR)': [fallout], '특이성(Specificity, TNR)':[spe], 'RAS': [ras], 'f1_score':[f1]})

    # 오즈비
    coef = fit.params
    odds_rate = np.exp(coef)
    odds_rate_df = DataFrame(odds_rate, columns=['odds_rate'])
    
    #return (model, fit, summary, prs, cmdf, result_df, odds_rate_df)

    logit_result = LogitResult()
    logit_result.model = model
    logit_result.fit = fit
    logit_result.summary = summary
    logit_result.prs = prs
    logit_result.cmdf = cmdf
    logit_result.result_df = result_df
    logit_result.odds_rate_df = odds_rate_df

    return logit_result


#------------------------------
# 시계열데이터분석
#------------------------------
# 시계열 데이터 분석(차분을 수행해 최적의 결과를 유도)
def expTimeData(data, yname, sd_model="m", max_diff=1):
    plt.rcParams["font.family"] = 'AppleGothic' if sys.platform == 'darwin' else 'Malgun Gothic'
    plt.rcParams["font.size"] = 12
    plt.rcParams["axes.unicode_minus"] = False

    df = data.copy()

    # 데이터 정상성 여부
    stationarity = False

    # 반복 수행 횟수
    count = 0

    # 결측치 존재 여부
    na_count = df[yname].isna().sum()
    print("결측치 수: %d" % na_count)

    plt.figure(figsize=(4, 5))
    sb.boxplot(data=df, y=yname)
    plt.show()
    plt.close()
    
    # 시계열 분해
    model_name = 'multiplicative' if sd_model == 'm' else 'additive'
    sd = seasonal_decompose(df[yname], model=model_name)

    figure = sd.plot()
    figure.set_figwidth(15)
    figure.set_figheight(16)
    fig, ax1, ax2, ax3, ax4 = figure.get_children()
    figure.subplots_adjust(hspace=0.4)

    ax1.set_ylabel("Original")
    ax1.grid(True)
    ax1.title.set_text("Original")
    ax2.grid(True)
    ax2.title.set_text("Trend")
    ax3.grid(True)
    ax3.title.set_text("Seasonal")
    ax4.grid(True)
    ax4.title.set_text("Residual")

    plt.show()

    # ACF, PACF 검정
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))
    fig.subplots_adjust(hspace=0.4)

    sb.lineplot(data=df, x=df.index, y=yname, ax=ax1)
    ax1.title.set_text("Original")

    plot_acf(df[yname], ax=ax2)
    ax2.title.set_text("ACF Test")
        
    plot_pacf(df[yname], ax=ax3)
    ax3.title.set_text("PACF Test")
        
    plt.show()
    plt.close()

    while not stationarity:
        if count == 0:
            print("=========== 원본 데이터 ===========")
        else:
            print("=========== %d차 차분 데이터 ===========" % count)

        # ADF Test
        ar = adfuller(df[yname])

        ardict = {
            '검정통계량(ADF Statistic)': [ar[0]],
            '유의수준(p-value)': [ar[1]],
            '최적차수(num of lags)': [ar[2]],
            '관측치 개수(num of observations)': [ar[3]]   
        }

        for key, value in ar[4].items():
            ardict['기각값(Critical Values) %s' % key] = value

        stationarity = ar[1] < 0.05
        ardict['데이터 정상성 여부(0=False,1=True)'] = stationarity

        ardf = DataFrame(ardict, index=['ADF Test']).T

        print(tabulate(ardf, headers=["ADF Test", ""], tablefmt='psql', numalign="right"))

        # 차분 수행
        df = df.diff().dropna()

        # 반복을 계속할지 여부 판단
        count += 1
        if count == max_diff:
            break

# 시계열 데이터 분석(차분을 수행해 최적의 결과를 유도)
def exp_time_data(data, yname, sd_model="m", max_diff=1):
    expTimeData(data, yname, sd_model, max_diff)


#------------------------------
# 시계열데이터분석
#------------------------------
# 데이터 프레임의 인덱스를 datetime 형식으로 변환
def set_datetime_index(df, field=None, inplace=False):
    """
        데이터 프레임의 인덱스를 datetime 형식으로 변환
        Parameters
        -------
        - df: 데이터 프레임
        - inplace: 원본 데이터 프레임에 적용 여부
        Returns
        -------
        - 인덱스가 datetime 형식으로 변환된 데이터 프레임
    """

    if inplace:
        if field is not None:
            df.set_index(field, inplace=True)

        df.index = DatetimeIndex(df.index.values, freq=df.index.inferred_freq)
        df.sort_index(inplace=True)
    else:
        cdf = df.copy()

        if field is not None:
            cdf.set_index(field, inplace=True)

        cdf.index = DatetimeIndex(cdf.index.values, freq=cdf.index.inferred_freq)
        cdf.sort_index(inplace=True)
        return cdf
    


#------------------------------
# 머신러닝
#------------------------------
# 다항식 변환
def convertPoly(data, degree=2, include_bias=False):
    poly = PolynomialFeatures(degree=degree, include_bias=include_bias)
    fit = poly.fit_transform(data)
    x = DataFrame(fit, columns=poly.get_feature_names_out())
    return x

# 머신러닝 추세선 유도
def getTrend(x, y, degree=2, value_count=100):
    # degree=2 -> [ a, b, c ] ==> ax^2 + bx + c
    coeff = np.polyfit(x, y, degree)    #x에 따라 y값이 변하는 2차 방정식 생성

    if type(x) == 'list':   #x가 list 형식인 경우
        minx = min(x)
        maxx = max(x)
    else:
        minx = x.min()
        maxx = x.max()

    Vtrend = np.linspace(minx, maxx, value_count)   #평면좌표 상에서의 가상의 x값

    Ttrend = coeff[-1]  #생성된 x값에 따른 y값
    for i in range(0, degree):
        Ttrend += coeff[i] * Vtrend ** (degree - i)

    return (Vtrend, Ttrend)


'''
def regplot 변수설명
x_left : 왼쪽 그래프의 x 축 데이터
y_left : 왼쪽 그래프의 y 축 데이터
y_left_pred : 왼쪽 그래프에서의 예측된 y 값 데이터
값이 주어지면 예측된 데이터에 대한 산점도와 추세선이 함께 작성
left_title : 왼쪽 그래프의 제목
x_right : 오른쪽 그래프의 x 축 데이터
값이 주어지면 오른쪽 그래프 작성
y_right : 오른쪽 그래프의 y 축 데이터
y_right_pred : 오른쪽 그래프에서의 예측된 y 값 데이터
값이 주어지면 오른쪽 그래프에 예측된 데이터에 대한 산점도와 추세선 작성
right_title : 오른쪽 그래프의 제목
save_path : 그래프를 저장할 파일 경로
값이 주어지면 그래프를 해당 경로에 저장
getTrend() : 추세선을 계산하는 함수
x, y 데이터에 대한 추세선의 x, y 값을 반환
'''
def regplot(x_left, y_left, y_left_pred=None, left_title=None, x_right=None, y_right=None, y_right_pred=None, right_title=None, figsize=(10, 5), save_path=None):
    # x_left, y_left는 필수 입력 사항
    subcount = 1 if x_right is None else 2

    fig, ax = plt.subplots(1, subcount, figsize=figsize)

    axmain = ax if subcount == 1 else ax[0]

    # 왼쪽 산점도
    sb.scatterplot(x=x_left, y=y_left, label='data', ax=axmain)

    # 왼쪽 추세선
    x, y = getTrend(x_left, y_left)
    sb.lineplot(x=x, y=y, color='blue', linestyle='--', ax=axmain)

    # 왼쪽 추정치
    if y_left_pred is not None:
        sb.scatterplot(x=x_left, y=y_left_pred, label='predict', ax=axmain)
        # 추정치에 대한 추세선
        x,y = getTrend(x_left, y_left_pred)
        sb.lineplot(x=x, y=y, color='red', linestyle='--', ax=axmain)

    if left_title is not None:
        axmain.set_title(left_title)

    axmain.legend()
    axmain.grid()

    if x_right is not None:
        # 오른쪽 산점도
        sb.scatterplot(x=x_right, y=y_right, label='data', ax=ax[1])
        
        # 오른쪽 추세선
        x, y = getTrend(x_right, y_right)
        sb.lineplot(x=x, y=y, color='blue', linestyle="--", ax=ax[1])
    
        # 오른쪽 추정치
        if y_right_pred is not None:
            sb.scatterplot(x=x_right, y=y_right_pred, label='predict', ax=ax[1])
            # 추정치에 대한 추세선
            x, y = getTrend(x_right, y_right_pred)
            sb.lineplot(x=x, y=y, color='red', linestyle="--", ax=ax[1])
        
        if right_title is not None:
            ax[1].set_title(right_title)
            
        ax[1].legend()
        ax[1].grid()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=300)
        
    plt.show()
    plt.close()


# 머신러닝(회귀분석-지도학습)
'''
class OlsResult 갱신, class RegMetric 추가
ml_ols 모듈 설명
xnames에 독립변수 이름(들) 입력- list, 문자열
yname에 종속변수 이름
degree로 차수(1-단순선형회귀, 2-다항회귀)
test_size로 데이터 분할(train/test)
random_state로 학습 데이터 조합 설정 가능(데이터 분할 고정)
'''
def ml_ols(data, xnames, yname, degree=1, test_size=0.25, use_scalling=False, random_state=777):
    # 표준화 설정이 되어 있다면 표준화 수행
    if scalling:
        data = scalling(data)

    # 독립변수 이름이 문자열로 전달되었다면 콤마 단위로 잘라서 리스트로 변환
    # 띄어쓰기 금지 ex) xnames="길이,높이,두께" -> True / xnames="길이, 높이, 두께" -> False
    if type(xnames) == str:
        xnames = xnames.split(',')

    # 독립변수 추출
    x = data.filter(xnames)

    # 종속변수 추출 - 1차원 y = data.filter([yname])은 2차원 DF 형태
    y = data[yname]

    # 2차식 이상으로 설종된 경우 차수에 맞게 변환
    if degree > 1:
        x = convertPoly(x, degree=degree)

    # 데이터 분할 비율이 0보다 크면 분할 수행
    if test_size > 0:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)
    else:
        x_train = x
        y_train = y
        x_test = None
        y_test = None

    # 회귀분석 수행
    model = LinearRegression()
    fit = model.fit(x_train, y_train)

    result = OlsResult()
    result.model = model
    result.fit = fit
    result.coef = fit.coef_ #계수
    result.intercept = fit.intercept_   #절편

    result.x_train = x_train.copy()
    result.y_train = y_train.copy()
    result.train_pred = result.fit.predict(result.x_train)

    if x_test is not None and y_test is not None:
        result.x_test = x_test.copy()
        result.y_test = y_test.copy()
        result.test_pred = result.fit.predict(result.x_test)
        result.setRegMetric(y_train, result.train_pred, y_test, result.test_pred)
    else:
        result.setRegMetric(y_train, result.train_pred)

    # 결과표 함수 호출
    x_train[yname] = y_train
    result.table = get_ols_table(x_train, xnames, yname, result.intercept, result.coef, result.train_pred)

    return result


# 예측값을 위한 predict
def get_ols_table(data, xnames, yname, intercept, coef, predict):
    # 독립변수 이름이 문자열로 전달되면 콤마 단위로 잘라서 리스트로 변화
    # # 띄어쓰기 금지 ex) xnames="길이,높이,두께" -> True / xnames="길이, 높이, 두께" -> False
    if type(xnames) == str:
        xnames = xnames.split(',')

    # 독립변수 추출
    x = data.filter(xnames)

    # 종속변수 추출 - 1차원 y = data.filter([yname])은 2차원 DF 형태
    y = data[yname]

    # 절편과 계수를 하나의 배열로 결합
    params = np.append(intercept, coef)

    # 상수항 추가
    designX = x.copy()
    designX.insert(0, '상수', 1)

    # 행렬곱 구하기
    dot = np.dot(designX.T,designX)

    # 행렬곱에 대한 역행렬
    inv = np.linalg.inv(dot)

    # 역행렬의 대각선 반환
    dia = inv.diagonal()

    # 평균 제곱오차
    MSE = (sum((y-predict)**2)) / (len(designX)-len(designX.iloc[0]))

    # 표준오차
    se_b = np.sqrt(MSE * dia)

    # t 값
    ts_b = params / se_b

    # p값
    p_values = [2*(1-stats.t.cdf(np.abs(i),(len(designX)-len(designX.iloc[0])))) for i in ts_b]
    
    # VIF
    vif = []

    # 훈련데이터에 대한 독립/종속변수를 결합한 완전한 DF 준비
    data = x.copy()
    data[yname] = y
    # print(data)
    # print("-"*50)

    # 다중 공선성 계산을 위한 VIF 생성 
    for i, v in enumerate(x.columns):
        j = list(data.columns).index(v) #행의 index 정보 추출
        vif.append(variance_inflation_factor(data, j))  #VIF를 계산하고, vif 리스트에 추가

    # 결과표 구성
    table = DataFrame({
        "종속변수": [yname] * len(x.columns),
        "독립변수": x.columns,
        "B": coef,
        "표준오차": se_b[1:],
        "β": 0,
        "t": ts_b[1:],
        "유의확률": p_values[1:],
        "VIF": vif,
    })

    return table


# 선형회귀 모델의 다항회귀 모델화 - sklearn의 PolynomialFeatures
# 훈련/검증 데이터의 손실률, 절대오차 그래프 시각화
def tf_result_plot(result, figsize=(15, 5), dpi=150):
    # 학습 결과에 대한 DF 생성
    result_df = DataFrame(result.history)
    result_df['epochs'] = result_df.index+1
    result_df.set_index('epochs', inplace=True)

    # 학습 결과 그래프의 컬럼 명
    column_names = result_df.columns

    # 학습데이터에 대한 필드이름
    train_column_name = [column_names[0], column_names[1]]

    # 검증데이터에 대한 필드이름
    test_column_name = [column_names[2], column_names[3]]

    # 학습 결과 그래프

    # # 그래프 객체 생성
    fig, ax = plt.subplots(1,2, figsize=figsize, dpi=dpi)

    # # 훈련 및 검증 데이터의 손실률, 절대오차 그래프 그리기
    for i, v in enumerate(ax):
        sb.lineplot(x=result_df.index, y=train_column_name[i], data=result_df, color='blue', label=train_column_name[i], ax=v)
        sb.lineplot(x=result_df.index, y=test_column_name[i], data=result_df, color='orange', label=test_column_name[i], ax=v)
        v.set_title(train_column_name[i])
        v.set_xlabel('ephocs')
        v.set_ylabel(train_column_name[i])
        v.grid()
        v.legend()
        
    plt.show()
    plt.close()

    return result_df


# 로지스틱 회귀 결과 검정
def tf_logit_result(model, fit, x, y):    
    # 예측값 생성
    pred_bool = model.predict(x).flatten() > 0.5
    pred = pred_bool.astype(int)
    
    # 혼동행렬
    cm = confusion_matrix(y, pred)
    tn, fp, fn, tp = cm.ravel()
    cmdf = DataFrame([[tn, fn], [fp, tp]], index=['True', 'False'], columns=['Negative', 'Positive'])

    # RAS
    ras = roc_auc_score(y, pred)

    # 위양성율, 재현율, 임계값(사용안함)
    fpr, tpr, thresholds = roc_curve(y, pred)

    # 정확도
    acc = accuracy_score(y, pred)

    # 정밀도
    pre = precision_score(y, pred)

    # 재현율
    recall = recall_score(y, pred)

    # F1 score
    f1 = f1_score(y, pred)

    # 위양성율
    fallout = fp / (fp + tn)

    # 특이성
    spe = 1 - fallout

    result_df = DataFrame({'정확도(Accuracy)':[acc], '정밀도(Precision)':[pre], '재현율(Recall, TPR)':[recall], '위양성율(Fallout, FPR)': [fallout], '특이성(Specificity, TNR)':[spe], 'RAS': [ras], 'f1_score':[f1]})

    # 모델 가중치와 편향 얻기
    weights, bias = model.layers[1].get_weights()
    
    # 오즈비 계산
    odds_ratio = np.exp(weights[0])

    logit_result = LogitResult()
    logit_result.model = model
    logit_result.fit = fit
    logit_result.summary = model.summary()
    #logit_result.prs = prs
    logit_result.cmdf = cmdf
    logit_result.result_df = result_df
    logit_result.odds_rate_df = odds_ratio
    
    return logit_result


'''
모듈설명
- 데이터를 학습하고 성능을 평가
- 모델의 성능을 교차 검증을 통해 안정적으로 측정
- 결과를 분석 및 시각화
> 머신러닝 모델 성능 평가
F. 데이터 마이닝\02. Sklearn\모듈화.ipynb 참고
'''
# ML성능1
# **kargs: 다양한 하이퍼파라미터(옵션)들을 받을 수 있는 가변 인자
def singleML(modelName, x, y=None, cv=5, **kargs):
    # 모델 생성
    model = modelName(**kargs)
    # 교차 검증
    score = cross_val_score(model, x, y, cv=cv).mean()
    # 결과 DF
    df = DataFrame(cross_validate(model, x, y, cv=cv))
    return [model, score, df]

'''
두 번째 모듈
GridSearchCV : 머신러닝에서 모델의 성능향상을 위해 쓰이는 기법중 하나. 
사용자가 직접 모델의 하이퍼 파라미터의 값을 리스트로 입력하면 
값에 대한 경우의 수마다 예측 성능을 측정 평가/비교하면서 
최적의 하이퍼 파라미터 값을 찾는 과정을 진행.
'''
# # ML성능2
# **kargs: 다양한 하이퍼파라미터(옵션)들을 받을 수 있는 가변 인자
def gridML(modelName, x, y=None, params={}, cv=5, **kargs):
    # 모델 생성
    model = modelName(**kargs)
    # grid 생성
    grid = GridSearchCV(model, param_grid=params, cv=cv)

    try:
        grid.fit(x,y)   # 지도학습
    except:
        grid.fit(x) # 비지도학습

    result_df = DataFrame(grid.cv_results_["params"])
    result_df['mean_test_score'] = grid.cv_results_['mean_test_score']
    result_df.sort_values(by='mean_test_score', ascending=False)
    
    return [grid.best_estimator_, grid.best_score_, result_df]

