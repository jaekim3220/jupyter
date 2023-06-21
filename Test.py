# 1번
f = open ("Hello.txt", "w", encoding="utf-8")
f.write("Hello World\n")
f.write("Hello World\n")
f.write("Hello World\n")
f.write("Hello World\n")
f.close()

# 2번
with open ("Hello.txt", "w", encoding="utf-8") as f:
    f.write("Hello World\n")
    f.write("Hello World\n")
    f.write("Hello World\n")
    f.write("Hello World\n")

# %%
f = open ("Hello.txt", "r", encoding="utf-8")
content = f.read()
print(content)
f.close()
#%% 기억나지 않는 temp 기능을 복습
list = [1, 2]
temp = list[0]  # 임시 변수에 list[0]의 값을 저장
list[0] = list[1]  # list[0]의 값을 list[1]으로 변경
list[1] = temp  # list[1]의 값을 임시 변수(temp)의 값으로 변경
print(list)
