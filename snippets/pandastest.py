import pandas as pd


def main():
    titanic = pd.read_csv('titanic.csv')
    print(titanic)
    print(titanic.head(10))
    print(titanic.dtypes)
    titanic.to_excel("titanic.xlsx", sheet_name="passengers", index=False)

    print('Shape', titanic.shape)
    print('Shape (np)', titanic.values.shape)

    print(titanic.values)

    age = titanic['Age']
    print(type(titanic), type(age))
    age_sex = titanic[['Age', 'Sex']]
    print(type(titanic), type(age_sex))

    print(titanic[titanic['Sex'] == 'male']['Name'])
    print('Male average age  ', titanic[titanic['Sex'] == 'male']['Age'].mean())
    print('Female average age', titanic[titanic['Sex'] == 'female']['Age'].mean())
    print('Male survived average age  ', titanic[(titanic['Sex'] == 'male') & (titanic['Survived']== 1)]['Age'].mean())
    print('Male died average age      ', titanic[(titanic['Sex'] == 'male') & (titanic['Survived'] == 0)]['Age'].mean())


if __name__ == '__main__':
    main()

