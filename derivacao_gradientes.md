# Derivação dos gradientes — APS Parte 1 (RNN/BPTT)

> Use este arquivo como referência durante a transcrição manuscrita. Cada bloco corresponde a uma página do papel.

---

## Setup (escreva no topo da primeira folha)

### Arquitetura

Para $t = 3, 4, \ldots, n$:

$$h_{t,1} = \tanh(a\, y_{t-2} + c)$$

$$h_{t,2} = \tanh(a\, y_{t-1} + b\, h_{t,1} + c)$$

$$\hat{y}_t = u\, h_{t,2} + v$$

### Perda

$$L_t = \frac{1}{2}(y_t - \hat{y}_t)^2$$

$$L = \sum_{t=3}^{n} L_t$$

### Identidade da tanh

$$\frac{d}{dx}\tanh(x) = 1 - \tanh^2(x)$$

Em particular, $\dfrac{\partial h_{t,1}}{\partial s_1} = 1 - h_{t,1}^2$ e $\dfrac{\partial h_{t,2}}{\partial s_2} = 1 - h_{t,2}^2$, onde:

$$s_1 := a\, y_{t-2} + c$$

$$s_2 := a\, y_{t-1} + b\, h_{t,1} + c$$

---

## Preliminar — derivada da perda em relação à saída

Como $L_t = \dfrac{1}{2}(y_t - \hat{y}_t)^2$, derivamos em relação a $\hat{y}_t$ (pela regra da cadeia simples, com fator interno $-1$ porque $\hat{y}_t$ entra com sinal negativo dentro do quadrado):

$$\frac{\partial L_t}{\partial \hat{y}_t} = 2 \cdot \frac{1}{2} \cdot (y_t - \hat{y}_t) \cdot (-1) = -(y_t - \hat{y}_t)$$

Reescrevendo trocando o sinal:

$$\boxed{\;\frac{\partial L_t}{\partial \hat{y}_t} = \hat{y}_t - y_t\;}$$

> **Atenção ao sinal**: o $(-1)$ vem da derivada interna de $(y_t - \hat{y}_t)$ em relação a $\hat{y}_t$. Como $y_t$ é constante (alvo) e $\hat{y}_t$ aparece com coeficiente $-1$, sua derivada parcial é $-1$. Esse sinal **inverte** o termo $(y_t - \hat{y}_t)$ para $(\hat{y}_t - y_t)$. É por isso que no código aparece `err = y_hat - y[2:]`.

---

## (1) Gradiente em relação a $v$

$v$ aparece **apenas** em $\hat{y}_t = u\, h_{t,2} + v$, somando linearmente.

Pela regra da cadeia:

$$\frac{\partial L_t}{\partial v} = \frac{\partial L_t}{\partial \hat{y}_t} \cdot \frac{\partial \hat{y}_t}{\partial v}$$

Como $\dfrac{\partial \hat{y}_t}{\partial v} = 1$:

$$\boxed{\;\frac{\partial L_t}{\partial v} = (\hat{y}_t - y_t)\;}$$

---

## (2) Gradiente em relação a $u$

$u$ aparece **apenas** em $\hat{y}_t = u\, h_{t,2} + v$, multiplicando $h_{t,2}$.

Pela regra da cadeia:

$$\frac{\partial L_t}{\partial u} = \frac{\partial L_t}{\partial \hat{y}_t} \cdot \frac{\partial \hat{y}_t}{\partial u} = (\hat{y}_t - y_t) \cdot h_{t,2}$$

$$\boxed{\;\frac{\partial L_t}{\partial u} = (\hat{y}_t - y_t)\, h_{t,2}\;}$$

---

## (3) Gradiente em relação a $b$

$b$ aparece **apenas** em $s_2 = a\, y_{t-1} + b\, h_{t,1} + c$ (e portanto dentro de $h_{t,2}$). O caminho até $L_t$ é:

$$b \longrightarrow s_2 \longrightarrow h_{t,2} \longrightarrow \hat{y}_t \longrightarrow L_t$$

Aplicando a regra da cadeia em quatro elos:

$$\frac{\partial L_t}{\partial b} = \frac{\partial L_t}{\partial \hat{y}_t} \cdot \frac{\partial \hat{y}_t}{\partial h_{t,2}} \cdot \frac{\partial h_{t,2}}{\partial s_2} \cdot \frac{\partial s_2}{\partial b}$$

Cada fator:

- $\dfrac{\partial L_t}{\partial \hat{y}_t} = \hat{y}_t - y_t$
- $\dfrac{\partial \hat{y}_t}{\partial h_{t,2}} = u$
- $\dfrac{\partial h_{t,2}}{\partial s_2} = 1 - h_{t,2}^2$
- $\dfrac{\partial s_2}{\partial b} = h_{t,1}$

Multiplicando:

$$\boxed{\;\frac{\partial L_t}{\partial b} = (\hat{y}_t - y_t) \cdot u \cdot (1 - h_{t,2}^2) \cdot h_{t,1}\;}$$

---

## Observação — definição do fator auxiliar $\Delta_t$

Os três primeiros fatores $(\hat{y}_t - y_t) \cdot u \cdot (1 - h_{t,2}^2)$ aparecem em **toda** derivada cujo caminho atravessa $h_{t,2}$. Por economia de notação, defina:

$$\boxed{\;\Delta_t := (\hat{y}_t - y_t) \cdot u \cdot (1 - h_{t,2}^2)\;}$$

Com isso, $\dfrac{\partial L_t}{\partial b} = \Delta_t \cdot h_{t,1}$.

---

## (4) Gradiente em relação a $a$ — cadeia dupla

$a$ aparece em **dois lugares**:

1. **Diretamente** em $s_2$ via o termo $a\, y_{t-1}$.
2. **Indiretamente** em $s_2$ via $b\, h_{t,1}$, pois $h_{t,1}$ também depende de $a$ (já que $s_1 = a\, y_{t-2} + c$).

### Esquema dos caminhos

```
                   y_{t-1}
                      │
        ┌─── (direto) ┘
   a ───┤
        └─── (indireto) ─── y_{t-2} → s_1 → h_{t,1} ─── s_2 ── h_{t,2} ── ŷ_t ── L_t
                                                       ↑
                                                       │  (caminho direto também chega aqui)
```

### Cadeia geral até $s_2$

$$\frac{\partial L_t}{\partial a} = \frac{\partial L_t}{\partial \hat{y}_t} \cdot \frac{\partial \hat{y}_t}{\partial h_{t,2}} \cdot \frac{\partial h_{t,2}}{\partial s_2} \cdot \frac{\partial s_2}{\partial a}$$

Os três primeiros fatores formam o $\Delta_t$:

$$\frac{\partial L_t}{\partial a} = \Delta_t \cdot \frac{\partial s_2}{\partial a}$$

### Cálculo de $\dfrac{\partial s_2}{\partial a}$ (parte com cadeia dupla)

Como $s_2 = a\, y_{t-1} + b\, h_{t,1} + c$, e tanto $a\, y_{t-1}$ quanto $h_{t,1}$ dependem de $a$:

$$\frac{\partial s_2}{\partial a} = \underbrace{y_{t-1}}_{\text{caminho direto}} + \underbrace{b \cdot \frac{\partial h_{t,1}}{\partial a}}_{\text{caminho indireto}}$$

### Cálculo de $\dfrac{\partial h_{t,1}}{\partial a}$

Como $h_{t,1} = \tanh(s_1)$ e $s_1 = a\, y_{t-2} + c$:

$$\frac{\partial h_{t,1}}{\partial a} = \frac{\partial h_{t,1}}{\partial s_1} \cdot \frac{\partial s_1}{\partial a} = (1 - h_{t,1}^2) \cdot y_{t-2}$$

### Substituição de baixo para cima

$$\frac{\partial s_2}{\partial a} = y_{t-1} + b \cdot (1 - h_{t,1}^2) \cdot y_{t-2}$$

$$\frac{\partial L_t}{\partial a} = \Delta_t \cdot \left[\, y_{t-1} + b\,(1 - h_{t,1}^2)\, y_{t-2}\, \right]$$

### Forma expandida (a que o enunciado pede)

$$\boxed{\;\frac{\partial L_t}{\partial a} = (\hat{y}_t - y_t) \cdot u \cdot (1 - h_{t,2}^2) \cdot \left[\, y_{t-1} + b\,(1 - h_{t,1}^2)\, y_{t-2}\, \right]\;}$$

---

## (5) Gradiente em relação a $c$ — segunda cadeia dupla

$c$ aparece em **dois lugares**, analogamente a $a$:

1. **Diretamente** em $s_2$ via o termo "$+ c$" final.
2. **Indiretamente** em $s_2$ via $h_{t,1}$, pois $s_1 = a\, y_{t-2} + c$ contém $c$.

### Cadeia até $s_2$

$$\frac{\partial L_t}{\partial c} = \Delta_t \cdot \frac{\partial s_2}{\partial c}$$

### Cálculo de $\dfrac{\partial s_2}{\partial c}$

$$\frac{\partial s_2}{\partial c} = \underbrace{1}_{\text{caminho direto}} + \underbrace{b \cdot \frac{\partial h_{t,1}}{\partial c}}_{\text{caminho indireto}}$$

### Cálculo de $\dfrac{\partial h_{t,1}}{\partial c}$

$$\frac{\partial h_{t,1}}{\partial c} = \frac{\partial h_{t,1}}{\partial s_1} \cdot \frac{\partial s_1}{\partial c} = (1 - h_{t,1}^2) \cdot 1 = 1 - h_{t,1}^2$$

### Substituição

$$\frac{\partial s_2}{\partial c} = 1 + b\,(1 - h_{t,1}^2)$$

$$\frac{\partial L_t}{\partial c} = \Delta_t \cdot \left[\, 1 + b\,(1 - h_{t,1}^2)\, \right]$$

### Forma expandida

$$\boxed{\;\frac{\partial L_t}{\partial c} = (\hat{y}_t - y_t) \cdot u \cdot (1 - h_{t,2}^2) \cdot \left[\, 1 + b\,(1 - h_{t,1}^2)\, \right]\;}$$

---

## Quadro-resumo

Com $\Delta_t := (\hat{y}_t - y_t) \cdot u \cdot (1 - h_{t,2}^2)$:

| Parâmetro | $\dfrac{\partial L_t}{\partial \cdot}$ |
|---|---|
| $u$ | $(\hat{y}_t - y_t)\, h_{t,2}$ |
| $v$ | $(\hat{y}_t - y_t)$ |
| $a$ | $\Delta_t \cdot \left[\, y_{t-1} + b\,(1 - h_{t,1}^2)\, y_{t-2}\, \right]$ |
| $b$ | $\Delta_t \cdot h_{t,1}$ |
| $c$ | $\Delta_t \cdot \left[\, 1 + b\,(1 - h_{t,1}^2)\, \right]$ |

---

## Gradiente total (full-batch)

Como $L = \sum_{t=3}^{n} L_t$, e a derivada de uma soma é a soma das derivadas:

$$\nabla L = \sum_{t=3}^{n} \nabla L_t$$

Para cada parâmetro $\theta \in \{u, v, a, b, c\}$:

$$\frac{\partial L}{\partial \theta} = \sum_{t=3}^{n} \frac{\partial L_t}{\partial \theta}$$

Estas somas são realizadas no código numpy via `np.sum(...)`.

---

## Sobre os sinais (resumo dos pontos de confusão)

| Onde | O que aparece | Por quê |
|---|---|---|
| Derivada de $L_t$ em relação a $\hat{y}_t$ | $+(\hat{y}_t - y_t)$ | A derivada de $(y_t - \hat{y}_t)^2$ em relação a $\hat{y}_t$ é $2(y_t - \hat{y}_t) \cdot (-1) = -2(y_t - \hat{y}_t)$. Com o fator $\tfrac{1}{2}$ da perda, sobra $-(y_t - \hat{y}_t) = (\hat{y}_t - y_t)$. |
| Caminho direto de $a$ em $s_2$ | $+\, y_{t-1}$ | $\partial(a\, y_{t-1})/\partial a = y_{t-1}$, sem sinal. |
| Caminho indireto de $a$ via $h_{t,1}$ | $+\, b\,(1 - h_{t,1}^2)\, y_{t-2}$ | $\partial s_2/\partial h_{t,1} = b$ (positivo), e $\partial h_{t,1}/\partial a = (1-h_{t,1}^2)\,y_{t-2}$ (também positivo). Soma com o caminho direto. |
| Derivada da tanh | $+(1 - h^2)$ | Sempre positiva (entre 0 e 1), pois $h \in (-1, 1)$. Nunca aparece com sinal negativo. |
| Caminho direto de $c$ em $s_2$ | $+1$ | $\partial(c)/\partial c = 1$. |
| Caminho indireto de $c$ via $h_{t,1}$ | $+\, b\,(1 - h_{t,1}^2)$ | Mesma estrutura de $a$, mas sem o $y_{t-2}$ (porque $\partial s_1/\partial c = 1$, não $y_{t-2}$). |

**Regra geral**: nenhum sinal negativo aparece nas derivadas — todos os termos somam positivamente. O único sinal a ter cuidado é o $(\hat{y}_t - y_t)$ no início, que vem da derivada da perda quadrática.

---

## Conferência rápida (substitua os valores numéricos do treinamento para checar)

Após o treinamento, os valores finais foram aproximadamente:

- $u \approx 1{,}3639$
- $v \approx 5{,}3723$
- $a \approx 0{,}8500$
- $b \approx -2{,}2916$
- $c \approx -2{,}3148$

Para um $t$ qualquer, pegue os valores correspondentes de $h_{t,1}, h_{t,2}, \hat{y}_t, y_t$ do código e verifique que as cinco fórmulas reproduzem os valores que `np.sum(...)` produz no notebook. Esse é o teste numérico de sanidade.
