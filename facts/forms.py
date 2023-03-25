from django import forms

class factsForm(forms.Form):
    facts = forms.CharField(max_length=10000)