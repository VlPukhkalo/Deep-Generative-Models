# HW3 отчет

## Проекции изображений в скрытое пространство 

Выбран метод, в котором начальная точка берется случайно. Добавлен шедулер, гиперпараметры подбираются с помощью случайного поиска по сетке. 
Оказалось, что квази-оптимальные гиперпараметры во всех случаях совпали: 
{'num_steps': 150, 'initial_learning_rate': 0.05, 'regularize_noise_weight': 500000.0, 'rec_weight': 0.5, 'lpips_weight': 1}

## Style transfer

Из латентного вектора взяты компоненты с 9 по 17: те, которые отвечают за освещение, текстуру и прочие фоновые вещи. 
Эти компоненты подменялись значениями из спроецированного в лат.прос-во изображения "стиля" с коэффициентом psi=0.1. 

## Expression Transfer

Собран набор из 3 эмоций и нейрального лица. Эмоция вычислялась вычитанием проекции нейтрального лица из проекции эмоционального лица. 
Затем получившийся вектор эмоции складывался с подобранными компонентами проекции героя. 
```
style = latent3 - latent2

for i in indeces:
    latent1[:, i] += style[:, i] * psi
```
Для каждой эмоции компоненты и параметр psi подбирался отдельно. 

```
if style == 'smile.jpg' or style == 'screem.jpg':
  result = interpolate_emotion(latent1, latent_neutral, latent2, psi=1, indeces = [4,5])
  result_image = result.detach().cpu().permute(0, 2, 3, 1).numpy()[0]
  row.append(result_image)
elif style == 'wtf.jpg':
  result = interpolate_emotion(latent1, latent_neutral, latent2, psi=1.3, indeces = [3,4,8,9])
  result_image = result.detach().cpu().permute(0, 2, 3, 1).numpy()[0]
  row.append(result_image)
```

## Face Swap

Добавлен новый лосс в фукнцию проекции лица в латентное пространство. 
Сохранен шедулер. Теперь стартовая точка – проекция лица в прост-во моделью e4e. 
Подобраны гиперпараметры:regularize_noise_weight=5e5, rec_weight=0.7, lpips_weight=0.5,face_weight = 1.1. 
