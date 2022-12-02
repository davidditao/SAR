function islr=islrfunc(x)

l=length(x);
a=find(x==max(x));
i=1;
for k=a-1:-1:2
    if(x(k)-x(k-1)<0&&x(k)-x(k+1)<0)
        lindian(i)=k;
        i=i+1;
    end
end
lindian1=max(lindian);

lindian=0;
i=1;
for k=a+1:l-1
    if(x(k)-x(k-1)<0&&x(k)-x(k+1)<0)
        lindian(i)=k;
        i=i+1;
    end
end
lindian2=min(lindian);

pmain=0;
for k=lindian1:lindian2
    pmain=pmain+x(k)^2;
end
x=x.^2;
ptotal=sum(x);
islr=10*log10((ptotal-pmain)/ptotal);
