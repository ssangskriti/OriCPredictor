
str = ['A', 'G', 'T', 'C']

p = []

i= 0

while i < len(str):
    ss = []
    ss += str[i]
    k = 0
    while k<len(str):
        sp = []
        sp += ss
        sp += str[k]
        n= 0
        while n < len(str):
            sp1 = []
            sp1 += sp
            sp1 += str[n]
            m = 0
            while m < len(str):
                sp2 = []
                sp2 += sp1
                sp2 += str[m]
                j = 0
                while j< len(str):
                    s = []
                    s+= sp2
                    s += (str[j])
                    #s= ''.join(s)
                    l = 0
                    while l< len(str):
                        sp3 = []
                        sp3 += s
                        sp3 += (str[l])
                        sp3= ''.join(sp3)

                        p.append(sp3)
                        l += 1
                    j += 1
                m +=1
            n+=1
        k += 1
    i = i+ 1


print (p)
