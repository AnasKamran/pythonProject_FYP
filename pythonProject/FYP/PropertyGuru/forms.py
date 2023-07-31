from django import forms
from .models import Feedback

class forms(forms.ModelForm):

    class Meta:
        model = Feedback
        fields = ['name', 'E_mail', 'ph_no','subject','Details']
        widgets = {
            'name': forms.TextInput(attrs={'class': 'form-control'}),
            'E_mail': forms.EmailInput(attrs={'class': 'form-control'}),
            'ph_no': forms.NumberInput(attrs={'class': 'form-control'}),
            'subject': forms.TextInput(attrs={'class': 'form-control'}),
            'Details': forms.Textarea(attrs={'class': 'form-control form-controldetail'}),


        }




