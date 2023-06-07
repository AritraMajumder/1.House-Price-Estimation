def rangetofloat(x):
        switch = 0
        for i in x:
            if i=='P':
                switch = 1
            elif i=='S':
                switch = 2
            elif i=='-':
                switch = 3
            elif i=='A':
                switch = 4
        
        if switch==1:
            a = x.split('P')
            b = float(a[0])*272.25
            return float(b)
        elif switch==2:
            a = x.split('S')
            b = float(a[0])*10.7639
            return float(b)
        elif switch==3:
            a = x.split('-')
            b = (float(a[0])+float(a[-1]))/2
            return float(b)
        elif switch==4:
            a = x.split('A')
            b = float(a[0])*43560
            return float(b)    
        else:
            return float(x)
            

            
            
print(rangetofloat('21223Sqm'))

