from django import forms

class OptimalAIForm(forms.Form):
    # Helper function to create float-typed choice fields with Bootstrap classes
    @staticmethod

    def float_choice_field(label, choices):
        return forms.TypedChoiceField(
            label=label,
            choices=[(str(value), display) for value, display in choices],
            coerce=float,
            widget=forms.RadioSelect  # Optional: Use RadioSelect for better UI
        )

    # 이진 변수 (Binary Variables)
    SEX_CHOICES = [(0, 'Female'), (1, 'Male')]
    HYPERTENSION_CHOICES = [(0, 'No'), (1, 'Yes')]
    HYPERLIPIDEMIA_CHOICES = [(0, 'No'), (1, 'Yes')]
    SMOKING_CHOICES = [(0, 'No'), (1, 'Yes')]
    PREVIOUS_STROKE_CHOICES = [(0, 'No'), (1, 'Yes')]
    CORONARY_CHOLES_CHOICES = [(0, 'No'), (1, 'Yes')]
    ACTIVE_CANCER_CHOICES = [(0, 'No'), (1, 'Yes')]
    CONGESTIVE_HEART_FAILURE_CHOICES = [(0, 'No'), (1, 'Yes')]
    PERIPHERAL_ARTERY_CHO_CHOICES = [(0, 'No'), (1, 'Yes')]
    DIABETES_CHOICES = [(0, 'No'), (1, 'Yes')]
    ATRIAL_FIB_CHOICES = [(0, 'No'), (1, 'Yes')]
    ANTIPLATELET_CHOICES = [(0, 'No'), (1, 'Yes')]
    ANTICOAGULANT_CHOICES = [(0, 'No'), (1, 'Yes')]
    GROUP_CHOICES = [(0, 'Conventional'), (1, 'Intensive')]

    # Integer and Float Fields with Bootstrap classes
    pt_age = forms.IntegerField(
        label='Age',
        min_value=0,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Enter age'}),
        required=True
    )
    NIHSS_IAT_just_before = forms.FloatField(
        label='NIHSS Score',
        min_value=0,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Enter NIHSS score'}),
        required=True
    )
    Onset_to_registration_min = forms.FloatField(
        label='Onset to Registration (minutes)',
        min_value=0,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Enter onset to registration time'}),
        required=True
    )
    Systolic_enroll = forms.FloatField(
        label='SBP Enroll',
        min_value=0,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Enter SBP at enrollment'}),
        required=True
    )
    Hgb = forms.FloatField(
        label='Hemoglobin',
        min_value=0,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Enter hemoglobin level'}),
        required=True
    )
    WBC = forms.FloatField(
        label='White Blood Cell',
        min_value=0,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Enter WBC count'}),
        required=True
    )
    BMI = forms.FloatField(
        label='Body Mass Index',
        min_value=0,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Enter BMI'}),
        required=True
    )
    systolic_max = forms.FloatField(
        label='SBP Max',
        min_value=0,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Enter maximum SBP'}),
        required=True
    )
    systolic_min = forms.FloatField(
        label='SBP Min',
        min_value=0,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Enter minimum SBP'}),
        required=True
    )
    systolic_mean = forms.FloatField(
        label='SBP Mean',
        min_value=0,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Enter mean SBP'}),
        required=True
    )
    systolic_TR = forms.FloatField(
        label='SBP Time Rate',
        min_value=0,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Enter SBP time rate'}),
        required=True
    )
    systolic_SD = forms.FloatField(
        label='SBP Standard Deviation',
        min_value=0,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Enter SBP standard deviation'}),
        required=True
    )
    systolic_CV = forms.FloatField(
        label='SBP Coefficient of Variation',
        min_value=0,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Enter SBP CV'}),
        required=True
    )
    systolic_VIM = forms.FloatField(
        label='SBP Variation Independent of the Mean',
        min_value=0,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Enter SBP VIM'}),
        required=True
    )

    # Typed Choice Fields with Float Coercion and Bootstrap styling
    pt_sex = float_choice_field.__func__('Sex', SEX_CHOICES)
    HiBP = float_choice_field.__func__('Hypertension', HYPERTENSION_CHOICES)
    Hyperlipidemia = float_choice_field.__func__('Hyperlipidemia', HYPERLIPIDEMIA_CHOICES)
    Smoking = float_choice_field.__func__('Smoking', SMOKING_CHOICES)
    Previous_stroke_existence = float_choice_field.__func__('Previous Stroke', PREVIOUS_STROKE_CHOICES)
    CAOD합친것 = float_choice_field.__func__('CAOD', CORONARY_CHOLES_CHOICES)
    cancer_active = float_choice_field.__func__('Active Cancer', ACTIVE_CANCER_CHOICES)
    CHF_onoff = float_choice_field.__func__('Congestive Heart Failure', CONGESTIVE_HEART_FAILURE_CHOICES)
    PAOD_existence = float_choice_field.__func__('PAOD', PERIPHERAL_ARTERY_CHO_CHOICES)
    IV_tPA = float_choice_field.__func__('IV tPA', SMOKING_CHOICES)
    DM = float_choice_field.__func__('DM', DIABETES_CHOICES)
    A_fib합친것 = float_choice_field.__func__('Atrial Fibrillation', ATRIAL_FIB_CHOICES)
    Antiplatelet = float_choice_field.__func__('Antiplatelet', ANTIPLATELET_CHOICES)
    Anticoagulant = float_choice_field.__func__('Anticoagulant', ANTICOAGULANT_CHOICES)
    Group = float_choice_field.__func__('Group', GROUP_CHOICES)
