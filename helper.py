import numpy as np
from pandas import DataFrame
from math import sqrt
from scipy.stats import t
from sklearn.impute import SimpleImputer



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