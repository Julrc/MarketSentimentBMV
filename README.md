Este índice no debería usarse como una principal razón de compra; es solamente un indicador que, en conjunto con otros indicadores, puede ayudar en la toma de decisiones financieras. El índice combina distintos indicadores para crear un índice de sentimiento para la Bolsa Mexicana de Valores. Se utiliza un modelo de machine learning para evaluar el sentimiento en noticias y redes sociales.

1. Volatilidad 25%
Se calcula la volatilidad del IPC (^MXX), que mide las fluctuaciones en el mercado bursátil mexicano. Una mayor volatilidad puede indicar miedo en el mercado, mientras que una menor volatilidad puede sugerir confianza o avaricia.

2. Noticias 20%
Se obtienen las últimas noticias a través de Google RSS. Las noticias se limpian y se analizan con el modelo de sentimiento para asignarles un valor entre 0 y 1, donde 0 indica sentimiento negativo (miedo) y 1 indica sentimiento positivo (avaricia).

3. Redes sociales 20%
Se utiliza la red social Reddit para poder conseguir últimos posts en varios subreddits mexicanos sobre finanzas con ciertas palabras claves para obtener los últimos posts recientes. Estos posts pasan igualmente por el modelo de sentimiento y se les asigna un valor de entre 0 y 1.

4. Google Trends 10%
Se utilizan varias palabras clave de búsquedas en google y se calcula cuando la cantidad de búsquedas de ese tema tuvo un incremento significante en búsquedas. Por ejemplo un incremento de 200% de un dia a otro con el tema de ‘comprar acciones’ puede ser considerado una señal de avaricia en el mercado.

5. Momentum/Volumen 25%
Se analizan el momentum y el volumen de transacciones en el mercado para evaluar la fuerza y dirección de las tendencias actuales. Se comparan los volúmenes y el momentum actuales con los promedios de 30 y 90 días para detectar desviaciones significativas. Un aumento sustancial puede indicar avaricia, mientras que una disminución puede señalar miedo entre los inversores.
