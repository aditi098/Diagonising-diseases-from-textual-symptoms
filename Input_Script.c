#include<stdio.h>
#include<string.h>
#include<stdlib.h>
#include<time.h>

char sentences[7][30] ={ "I have ", "I am suffering from ", "I also have ", "I have been feeling ", "I have been suffering from ", "I feel ", "I think I have "};



//all symptoms
//char symptoms[20][30] = {"headache", "vomiting", "nausea", "bleeding gum", "itch", "rash", "fever", "diarrhea", "discomfort", "chest pain", "abdominal pain", "fatigue", "muscle pain", "chills", "eye pain", "joint pain","nerve pain","ligament pain","bleeding nose","tendon pain"};


//symptoms for typhoid
//char symptoms[10][30] = { "nausea", "bleeding gum", "itch", "fever", "muscle pain","eye pain", "joint pain","nerve pain","ligament pain","tendon pain"}; 

//symptoms for dengue
//char symptoms[8][30] = {"headache", "vomiting", "rash",  "diarrhea", "discomfort", "abdominal pain", "bleeding nose","nausea"};



//symptoms for malaria
char symptoms[9][30] = {"headache",  "fever", "diarrhea", "chest pain", "abdominal pain", "fatigue", "muscle pain", "chills", "joint pain"}; 


//symptoms for viral fever
//char symptoms[3][30] = {"headache",  "fever", "chills"};   



int main()
{

srand(time(0));
FILE *fptr= fopen("malaria.txt", "w");
for(int i=0;i<1000;i++)
    {
        //int list= {1,2,3};
        int ran= 1+rand()%3;
        //var patient= "Patient" + i + "\n";
        //fwrite(fh, patient);
        for(int j=0; j<ran; j++)
        {
            char sentence[30];
            strcpy(sentence,sentences[rand()%7]);
            char symptom1[20];
            strcpy(symptom1,symptoms[rand()%9]);
            char symptom2[20];
            do{
            strcpy(symptom2,symptoms[rand()%9]);
            }while(strcmp(symptom1, symptom2)==0);
            fprintf(fptr,"%s", sentence);
            fprintf(fptr, " %s", symptom1);
            fprintf(fptr, " %s", "and" );
            fprintf(fptr, " %s", symptom2 );
            fprintf(fptr, "%s", ". ");
        }
        fprintf(fptr, "%s", "\n");
    }

fclose(fptr);
return 0;
}
