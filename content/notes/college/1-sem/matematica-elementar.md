---
title: "Matemática Elementar"
date: 2023-03-28T08:28:34-04:00
draft: false
showToc: true
math: true
---
# Introdução
## Ementa
+ Números reais
+ Equações e inequações
+ Funções de uma variável real
+ Noções de trigonometria
## Bibliografia
+ Pré-calculo: Operações, Equações, Funções e Trigonometria (Francisco Magalhães Gomes)
+ Fundamentos de matemática elementar (somente 1 e 3)
## Avaliações
+ 3 avaliações
+ Tabalhos e listas de exercício
### Datas
+ [ ] P1 [20/04]
+ [ ] P2 [25/05]
+ [ ] P3 [22/06]
+ [ ] PO [29/06]

# Conjuntos Númericos
## Números naturais
+ Aqui começa do zero, mas note que não faz sentido para contar coisas (zero coisas?)
$\mathbb{N} = {0, 1, 2, 3...}$
+ Com ele não é possível obter números negativos, então a seguinte equação não possui solução
$X + 3 = 2$
## Números inteiros
$\mathbb{Z} = {..., -3, -2, -1, 0, +1, +2, +3, ...}$
+ Perceba que os naturais pertencem aos inteiros
### Adição de números de mesmo sinal
+ Soma e conserva o sinal. Ex:
$1+2 = +3\$
### Adição de números de sinais contrários
+ Subtraio e conservo o sinal maior. Ex:
$$-10+20 = +10$$
$$-30+20 = -10$$
### Multiplicação
$$(+1) \cdot (+1) = +1$$
$$(+1) \cdot (-1) = -1$$
$$(-1) \cdot (+1) = -1$$
$$(-1) \cdot (-1) = +1$$
+ Mas não conseguimos resolver
$$2x + 3 = 6$$
$$2x = 3$$
+ Então tem que criar um novo conjunto ksksk
## Racionais
### Demonstração de divisão de zero
$$\mathbb{R} = {\frac{a}{b}, a e b \in \mathbb{Z}, b \neq 0}$$
+ Dado C, o inverso é um número c', mas que $c \cdot c' = 1$
$$0 \cdot c' = 1 \nexists c$$
### Demonstração de números multiplicados por zero
$$0\cdot a = (0+0)a$$
$$0\cdot a = 0\cdot a + 0\cdot a$$
$$0 \in \mathbb{Z}, a \in \mathbb{Z}$$
$$0\cdot a \in \mathbb{Z}$$
$$\exists -0\cdot a \in \mathbb{Z}$$
$$0\cdot a - 0\cdot a= 0\cdot a + 0\cdot a - 0\cdot a$$
$$0 = 0\cdot a$$
### Somando frações
$$\frac{a}{b}+{c}{d} = \frac{ad+cb}{b\cdot d}$$
$$\frac{a}{b}+{c}{d} = \frac{ac}{b\cdot d}$$
+ Exemplo:
$$\frac{1}{2} + \frac{3}{4} = \frac{4+6}{8} = \frac{10}{8} = \frac{5}{4}$$
+ Toda fração admite ser irredutível (simplificar até o máximo) e é semelhante a uma fração iredutível
### Algumas regras
$$\frac{+1}{+1} = +1$$
$$\frac{+1}{-1} = -1$$
$$\frac{-1}{+1} = -1$$
$$\frac{-1}{-1} = +1$$
+ Mas esse conjunto também não resolve tudo
$$X^2 = 1^2 + 1^2$$
$$X^2 = 1 + 1$$
$$X^2 = 2$$
#### Usando redução ao absurdo
$$X^2 = 2, X \notin \mathbb{Q}$$
$$X \in \mathbb{Q}, X = \frac{a}{b}$$ (irredutível)
$$(\frac{a}{b})^2 = 2$$
$$\frac{a}{b}\cdot \frac{a}{b} = 2$$
$$\frac{a^2}{b^2} = 2$$
+ Se a^2 é par, a é par
$$a^2 = 2b^2$$
$$a = 2\cdot k, k \in \mathbb{Z}$$
$$(2\cdot k)^2 = 2\cdot b^2$$
$$2\cdot k\cdot 2\cdot k = 2b^2$$
$$4k^2 = 2b^2 \Rightarrow 2k^2 = b^2$$
+ Se $b^2$ é par, b é par
+ Dividir dois números pares resulta no menor elemento do conjunto que é impar, logo vai contra a/b serem irredutiveis
+ Absurdo!
### O que há nos números racionais
+ Inteiros
+ Representação decimal finita $\frac{1}{2} = 0,5$
+ Representação decimal infinita periódica $\frac{1}{3} = 0,33333...$
## Números reais
+ $\mathbb{R}$ = números reais
+ Possui:
+ Inteiros, representação finita, representação decimal infinita e periódica, representação decimal infinita não periódica ($\pi$)

# Leitura ()
+ Há autores que não consideram 0 zero parte do conjunto dos números naturais
+ Todo número racional pode ser representado pela divisão de dois números inteiros

# Aula 21/03
## Em $Q = {\frac{a}{b}, a, b \in Z, b \neq 0}$
+ Inteiros $\frac{a}{1}$
+ Representação decimal finita $\frac{1}{2} = 0,5$
+ Representação decimal infinita periódica $\frac{1}{3} = 0,333...$
## Transformação de números em representação decimal finita
$$0,37 = \frac{37}{100}$$
$$2,631 = \frac{2631}{1000}$$
## Transformação de números em representação decimal infinita
$$0,7777...$$
$$x= 0,7777...$$
$$10x = 7,7777....$$
+ Agora podemos subtrair
$$10x - x = 7,7777... - 0,7777.... = 9x = 7$$
$$x = \frac{7}{9}$$
+ Outro exemplo:
$$x = 6,434343...$$
$$100x = 643,434343...$$
+ Agora subtraimos
$$100x - x= 643,434343 - 6,434343 = 99x = 643 - 6 = 637$$
$$x = \frac{637}{99}$$
+ [ ] Hard mode:
$$x = 2,57919191...$$
$$100x = 257,919191...$$
+ Vamos usar o 100x como se fosse o próprio x nos exemplos anteriores:
$$100000x = 25791,919191...$$
$$100X = 257,919191...$$
$$100000x - 100x = 25791,919191 - 257,919191 = 9900x = 25534$$
$$x = \frac{25534}{9900}$$
### Curiosidade
$$0,9999... = 1$$
$$10x-x = 9,9999-0,9999 = 9x = 9$$
$$x = 1$$

## Mais frações
+ Sejam $a, b e c \in Z +q$
$$a = b\cdot c$$
+ b é um divisor de a
+ c é um divisor de a
$$b | a e c | a$$
+ A barrinha em cima signifca que divide, logo é divisor
+ a é um múltiplo de b
+ a é um múltiplo de c
+ múltiplos positivos $4 = {4, 8, 12, 16...} M(4)$
+ divisores positivos $4 = {1, 2, 4} = D(4)$
+ primos possuem apenas dois divisores positivos (primo e 1)
+ compostos: produto de primos
+ Todo número inteiro, ou é 1, -1, primo, ou composto
## T. Fundamental aritimética
+ Cada número pode ser escrito por produtos de números primos elevados ao máximo expoente
+ A partir dos primos a gente pode escrever qualquer número
## Existem infinitos números primos
+ Redução ao absurdo...
+ Suponha que $a_{1} = 2, a_{2} = 3, a_{3} = 5, a_{n}$ (que sejam finitos)
$$x = a_{1}\cdot a_{2}\cdot a_{3}\cdot a_{n} + 1$$
+ Esse x não pode ser 1, não pode ser primo, nem é composto
+ O erro está em supor que $a_{n}$ é o último primo
# Múltiplos e divisores comuns
+ mmc = mínimo múltiplo (positivo) comum entre 2 números
+ mdc = mínimo divisor comum entre 2 números:
+ 6 e 10
$$m(6) = {6, 12, 18, 24, 30,...}$$
$$m(10) = {10, 20, 30, 40,...}$$
$$mmc(6, 10) = 30$$
$$D(6) = {1, 2, 3, 6}$$
$$D(10) = {1, 2, 5, 10}$$
$$mdc(6, 10) = 2$$
## Observação
$$mmc(a,b) = \frac{a\cdot b}{mdc(a,b)}$$
## Regra prática
+ Ex 700, 120
+ Fatorar 120 e 700:
$$\frac{120}{2} = 60, \frac{60}{2} = 30, \frac{30}{2} = 15, \frac{15}{3} = 5, \frac{5}{5} = 1$$
$$\frac{700}{2} = 350, \frac{350}{2} = 175, \frac{175}{5} = 35, \frac{35}{5} = 7, \frac{7}{7} = 1$$
+ Para achar o mdc, pegue todos os resultados da fatoração em comum (2, 2, 5) e obtenha o produto. Este é o mdc.
$$mdc (120, 700) = 2\cdot 2\cdot 5 = 20$$
+ Para achar o mmc, escreva os resultados da fatoração em base comum, para o 120:
$$2^3 \cdot 3^1\cdot 5^1$$
+ Para o 700:
$$2^2 \cdot 5^2 \cdot 7^1$$
+ Copie a base que aparece nos dois e pegue o maior expoente:
$$mmc(120, 700) = 2^3 \cdot 3^1 \cdot 5^2 \cdot 7^1 = 4200$$
### Passo a passo
+ mdc = na fatoração, pegar somente os números que se repetem nos dois ao mesmo tempo
+ mmc = na fatoração, pegar todos os primos que aparecem e com o maior expoente

# Reais
+ O conjuto dos números reais é um corpo ordenado completo
+ Comutativo, associativo?
## Relação de ordem -> Vale a lei da tricotomia
## Domínio de integridade

# Exponenciação
## Expoentes naturais
+ Vale para os inteiros:
$$a^{0} = 1, a \neq 0$$
$$a^n = a\cdot a^{n-1}$$
$$a^n = a\cdot a\cdot a (n vezes)$$
$$a^m \cdot a^n = a^{m+n}$$
$$(a^m)^n = a^{m\cdot n}$$
+ Vale para os racionais:
$$a \geq 0$$
$$a^{\frac{1}{n}} = \sqrt{n}(a) = b tal que b^n = a$$
+ a tem que ser positivo porque quanto for elevado a 1/n vai ser positivo
+ se n for ímpar, a raiz vai ser negativa

# Razão
+ Razão é o quociente entre 2 números reais (dobro, triplo, metade, um terço)
+ Escala: distância desenho/distância real

# Proporção
+ Uma forma de comparar duas grandezas
## Proporção direta
$$\frac{a_{1}}{b_{1}} = \frac{a_{2}}{b_{2}}$$
## Proporção inversa
$$a_{1}\cdot b_{1} = a_{2}\cdot b_{2}$$
Ou:
$$\frac{a_{1}}{b_{2}} = \frac{a_{2}}{b_{1}}$$

# Conjuntos
## Conceitos primitivos
+ Conjunto, elementos de um conjunto, relação de pertinência
## Relações de ordem (parcial)
1. $A \subseteq A$ (reflexiva)
2. $A \subseteq B$ e $B \subseteq A$ então $\rightarrow A = B$ (antissimétrica)
3. $A \subseteq B$ e $B \subseteq C$ então $A \subseteq C$ (transitivo)

+ Dado conjunto A, $\wp (A) = {B  B \subseteq A}$. Ex:
$$A = {1, 2, 3}$$
$$P (A) = {\varnothing}, {1}, {2}, {3}, {1,2}, {1,3}, {2,3}, {1,2,3}$$

+ Se A tiver um número finito de elementos n, o número de elementos de P(A) é $2^n$
$$n = n(A)$$
$$n (p(A)) = 2^n$$

+ $A, B \subseteq U$
+ $A \cup B = {x / x \in A ou x \in B}$ -> União
+ $A \cap B = {x / x \in A e x \in B}$ -> Intersecção
+ $A \backslash B = A - B {x / x \in A e x \notin B}$ -> Diferença
+ $A^c = {x | x \notin A}$ Complementar de A

+ Quando $A \cap B = \varnothing$, dizemos que A e B são disjuntos. Propriedades
1. $A \subseteq B \rightarrow A \cup B = B$
2. $A \subseteq B \rightarrow A \cap B = A$
3. $A \cap (B \cup C) = (A \cap B) \cup (A \cap C)$
4. $A \cup (B \cap C) = (A \cup B) \cap (A \cup C)$

### Exercícios
+ Faça o 1, 2 e 4 acima, como o 3 demonstrao abaixo
+ Seja $x \in A \cap (B \cup C$. Então $x \in A$ e $x \in B \cup C$
+ Então $x \in A$ e $x \in B$ ou $x \in A$ e $x \in C$, logo $x \in (A \cap B) \cup (A \cap C)$
+ Agora faremos o contrário. Seja $x \in (A \cap B) \cup (A \cap C)$
+ $x \in A \cap B$ significa $x \in A e x \in B$, mesma coisa para $x \in A \cap C$
$$x \in A$$ e $$[x \in B ou x \in C]$$
$$x \in A$$ e $$[x \in B \cup C]$$
$$x \in A \cap (B \cup C)]$$

Sejam A, B conjuntos finitos, ou seja, A e B possuem um número finito de elementos. (caderno)
