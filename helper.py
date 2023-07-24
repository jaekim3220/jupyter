import numpy as np
from pandas import DataFrame, MultiIndex, concat
from math import sqrt
from scipy.stats import t, pearsonr, spearmanr
from sklearn.impute import SimpleImputer
from scipy.stats import shapiro, normaltest, ks_2samp, bartlett, fligner, levene, chi2_contingency
#--------------------------------------------------
# 모듈 불러오기
# import sys
# import os
# sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
# from helper import normality_test, equal_variance_test, independence_test, all_test
#--------------------------------------------------



def replaceMissingValue(df):
    imr = SimpleImputer(missing_values=np.nan, strategy='mean')
    df_imr = imr.fit_transform(df.values)
    re_df = DataFrame(df_imr, index=df.index, columns=df.columns)
    return re_df



def getIq(field):
    q1 = field.quantile(q=0.25)
    q3 = field.quantile(q=0.75)
    iqr = q3 - q1
    하한 = q1 - 1.5 * iqr
    상한 = q3 + 1.5 * iqr
    결측치경계 = [하한, 상한]
    return 결측치경계



def replaceOutlier(df, fieldName):
    cdf = df.copy()

    # fieldName이 List가 아니면 List로 변환
    if not isinstance(fieldName, list):
        fieldName = [fieldName]

    for f in fieldName:
        결측치경계 = getIq(cdf[f])
        cdf.loc[cdf[f] < 결측치경계[0], f] = np.nan
        cdf.loc[cdf[f] > 결측치경계[1], f] = np.nan

    return cdf



def setCategory(df, ignore=[]):
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

            # 제외목록이 있고, 해당 변수명이 제외목록에 포함되어 있으면 다음 변수로 이동
            if ignore and field_name in ignore:
                continue

            # 가져온 변수명에 대해 값의 종류별로 빈도를 카운트 한 후 인덱스 이름순으로 정렬
            vc = cdf[field_name].value_counts().sort_index()
            #print(vc)

            # 인덱스 이름순으로 정렬된 값의 종류별로 반복 처리
            for ii, vv in enumerate(list(vc.index)):
                # 일련번호값 생성
                vnum = ii + 1
                #print(vv, " -->", vnum)

                # 일련번호값으로 치환
                cdf.loc[cdf[field_name] == vv, field_name] = vnum

            # 해당 변수의 데이터 타입을 범주형으로 변환
            cdf[field_name] = cdf[field_name].astype('category')

    return cdf



def clearStopwords(nouns, stopwords_file_path="wordcloud/stopwords-ko.txt"):
    with open(stopwords_file_path, 'r', encoding='utf-8') as f:
        stopwords = f.readlines()
        
        for i, v in enumerate(stopwords):
            stopwords[i] = v.strip()

    data_set = []

    for v in nouns:
        if v not in stopwords:
            data_set.append(v)

    return data_set


# 신뢰구간 생성
def get_confidence_interval(data, clevel=0.95):
    n = len(data)                           # 샘플 사이즈
    dof = n - 1                             # 자유도
    sample_mean = data.mean()               # 표본 평균
    sample_std = data.std(ddof=1)           # 표본 표준 편차
    sample_std_error = sample_std / sqrt(n) # 표본 표준오차

    # 신뢰구간
    cmin, cmax = t.interval(clevel, dof, loc=sample_mean, scale=sample_std_error)
    
    return (cmin, cmax)


#--------------------------------------------------
# 분산분석(일원분산분석 (One-way ANOVA)) 모듈
#--------------------------------------------------

# 분산분석(일원분산분석 (One-way ANOVA)) 모듈2 -> 추가내용 : DataFrame 생성용
def normality_test(*any):
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

# 분산분석(일원분산분석 (One-way ANOVA)) 모듈1
def equal_variance_test(*any):
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

def independence_test(*any):
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

def all_test(*any):
    return concat([normality_test(*any), equal_variance_test(*any), independence_test(*any)])


#------------------------------
# 피어슨 상관분석
#------------------------------
def pearson_r(df):
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

        data.append({'fields': fields, 'statistic': s, 'pvalue': p, 'result': result})

    rdf = DataFrame(data)
    rdf.set_index('fields', inplace=True)
    
    return rdf
#------------------------------
# 스피어만 상관분석
#------------------------------
def spearman_r(df):
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

        data.append({'fields': fields, 'statistic': s, 'pvalue': p, 'result': result})

    rdf = DataFrame(data)
    rdf.set_index('fields', inplace=True)
    
    return rdf