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