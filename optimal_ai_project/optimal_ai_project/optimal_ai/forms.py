from django import forms

class OptimalAIForm(forms.Form):
    @staticmethod
    def float_choice_field(label, choices):
        return forms.TypedChoiceField(
            label=label,
            choices=[(str(value), display) for value, display in choices],
            coerce=float,
            required=True,
            widget=forms.RadioSelect(attrs={"class": "seg-radio"})
        )

    YESNO_CHOICES = [(0, "No"), (1, "Yes")]
    GROUP_CHOICES = [(0, "Conventional"), (1, "Intensive")]

    pt_age = forms.IntegerField(
        label="Age",
        min_value=0,
        required=True,
        widget=forms.NumberInput(attrs={"class": "form-control", "placeholder": "e.g., 68"})
    )
    NIHSS_IAT_just_before = forms.FloatField(
        label="NIHSS score",
        min_value=0,
        required=True,
        widget=forms.NumberInput(attrs={"class": "form-control", "placeholder": "e.g., 14"})
    )
    Hgb = forms.FloatField(
        label="Hemoglobin (Hgb)",
        min_value=0,
        required=True,
        widget=forms.NumberInput(attrs={"class": "form-control", "placeholder": "e.g., 13.2"})
    )
    systolic_min = forms.FloatField(
        label="SBP min",
        min_value=0,
        required=True,
        widget=forms.NumberInput(attrs={"class": "form-control", "placeholder": "e.g., 105"})
    )
    systolic_TR = forms.FloatField(
        label="SBP time rate",
        min_value=0,
        required=True,
        widget=forms.NumberInput(attrs={"class": "form-control", "placeholder": "e.g., 0.12"})
    )

    Group = float_choice_field.__func__("Group", GROUP_CHOICES)
    Hyperlipidemia = float_choice_field.__func__("Hyperlipidemia", YESNO_CHOICES)
    DM = float_choice_field.__func__("Diabetes mellitus (DM)", YESNO_CHOICES)
    Previous_stroke_existence = float_choice_field.__func__("Previous stroke", YESNO_CHOICES)
    Anticoagulant = float_choice_field.__func__("Anticoagulant use", YESNO_CHOICES)
