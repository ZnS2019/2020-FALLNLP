import numpy as np
add_price=1
replace_price=2

# 代价初始化
def price_init():
    replace_price = input('Enter the price of replacement, default 2 : ')
    add_price = input('Enter the price of add/delete, default 1 : ')
    if replace_price:
        replace_price = int(replace_price)
    else:
        replace_price = 2
    if add_price:
        add_price = int(add_price)
    else:
        add_price = 1

# 定义字符串距离类
class strDis:
    # 距离类初始化
    def __init__(self, str1, str2):
        self.str1 = str1
        self.str2 = str2
        self.len1 = len(str1)
        self.len2 = len(str2)
        self.dpTable = np.zeros((self.len1+1, self.len2+1))
        self.dpTable[1:self.len1+1, 0] = range(1, self.len1 + 1)
        self.dpTable[0, 1:self.len2+1] = range(1, self.len2 + 1)
        self.min_edit_distance = 0
        self.ptrTable = np.zeros((self.len1+1, self.len2+1))
        for i in range(self.len1+1):
            self.ptrTable[i,0]=2
        for j in range(self.len2+1):
            self.ptrTable[0,j]=1

    # 利用动态规划对字符串距离求解
    def dp(self):
        for i in range(1, self.len1+1):
            for j in range(1, self.len2+1):
                replace_dis = 0 if self.str1[i-1] == self.str2[j-1] else replace_price
                self.dpTable[i, j] = min(self.dpTable[i, j - 1] + add_price,
                                         self.dpTable[i - 1, j] + add_price,
                                         self.dpTable[i - 1, j - 1] + replace_dis)
                if self.dpTable[i, j - 1] + add_price == self.dpTable[i, j]:
                    self.ptrTable[i, j] = 1
                elif self.dpTable[i - 1, j] + add_price == self.dpTable[i, j]:
                    self.ptrTable[i, j] = 2
                elif replace_dis==0:
                    self.ptrTable[i, j] = 3
                else:
                    self.ptrTable[i, j] = 4
                # print(self.dpTable)
                # print(i,j,'\n\n')
    # 回溯得到路径
    def traceback(self):
        # print(self.dpTable)
        # print(self.ptrTable)
        print("字符串a和b的最小编辑距离是", self.dpTable[-1, -1])
        i,j = self.len1, self.len2
        while (i>0 or j>0):
            print()
            if self.ptrTable[i, j] == 1:
                print(self.str1[0:i], self.str2[0:j],"str1增加字符", self.str2[j-1])
                j = j-1
            elif self.ptrTable[i, j] == 2:
                print(self.str1[0:i], self.str2[0:j],"str1删减字符",self.str1[i-1])
                i = i-1
            elif self.ptrTable[i, j] == 3:
                print(self.str1[0:i], self.str2[0:j],"不变")
                i = i-1 if i>0 else 0
                j = j-1 if j>0 else 0
            elif self.ptrTable[i,j] == 4:
                print(self.str1[0:i], self.str2[0:j],"替换str1最后的字符")
                i = i-1 if i>0 else 0
                j = j-1 if j>0 else 0
            else:
                print(i,j)
                print(self.dpTable)
                print(self.ptrTable)
                break
            
        
def main():
    price_init()
    str1 = input("first string, default = intention")
    str2 = input("second string, default = excecution")
    if not str1:
        str1 = "intention"
    if not str2:
        str2 = "execution"
    p = strDis(str1, str2)
    p.dp()
    p.traceback()


if __name__ == '__main__':
    main()
