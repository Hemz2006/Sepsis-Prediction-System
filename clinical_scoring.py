import pandas as pd
from typing import Dict, Optional, Union, Tuple

class SepsisScoring:
    """
    Implementation of sepsis scoring systems: SOFA, qSOFA, and NEWS2
    Based on Sepsis-3 definitions and clinical criteria
    """
       # ---------- Compatibility Wrapper (FINAL FIX) ----------
    def calculate_septic_shock(
        self,
        map_mmhg: float,
        lactate: float = None,
        lactate_mmol_l: float = None,
        vasopressors: bool = None,
        on_vasopressors: bool = None,
        adequate_volume_resus: bool = True,
        sepsis_present: bool = False
    ):
        """
        Compatibility wrapper to safely handle Flask form + route arguments
        """

        # Resolve lactate value
        if lactate_mmol_l is None:
            lactate_mmol_l = lactate

        # Resolve vasopressor flag
        if on_vasopressors is None:
            on_vasopressors = vasopressors

        return self.assess_septic_shock(
            map_mmhg=map_mmhg,
            lactate_mmol_l=lactate_mmol_l,
            on_vasopressors=bool(on_vasopressors),
            adequate_volume_resus=bool(adequate_volume_resus),
            sepsis_present=bool(sepsis_present)
        )


    def __init__(self):
        self.sofa_criteria = self._initialize_sofa_criteria()
    
    def _initialize_sofa_criteria(self) -> Dict:
        """Initialize SOFA scoring criteria for each organ system"""
        return {
            'respiration': {
                'name': 'PaO2/FiO2 (mmHg)',
                'ranges': [(400, float('inf'), 0), (300, 399, 1), (200, 299, 2), (100, 199, 3), (0, 99, 4)],
                'respiratory_support_needed': [False, False, False, True, True]
            },
            'coagulation': {
                'name': 'Platelets (×10³/μL)',
                'ranges': [(150, float('inf'), 0), (100, 149, 1), (50, 99, 2), (20, 49, 3), (0, 19, 4)]
            },
            'liver': {
                'name': 'Bilirubin (mg/dL)',
                'ranges': [(0, 1.2, 0), (1.2, 1.9, 1), (2.0, 5.9, 2), (6.0, 11.9, 3), (12.0, float('inf'), 4)]
            },
            'cardiovascular': {
                'name': 'MAP and vasopressor requirements',
                'criteria': 'complex'
            },
            'cns': {
                'name': 'Glasgow Coma Scale',
                'ranges': [(15, 15, 0), (13, 14, 1), (10, 12, 2), (6, 9, 3), (0, 5, 4)]
            },
            'renal': {
                'name': 'Creatinine (mg/dL) or urine output',
                'criteria': 'complex'
            }
        }
    
    # ---------- SOFA Methods ----------

# ---------- Respiration ----------
    def calculate_sofa_respiration(self,pao2_fio2: float, respiratory_support: bool = False) -> int:
        """
        PaO2/FiO2 in mmHg.
        Respiratory support must be True for scores 3 or 4.
        """
        if pao2_fio2 >= 400:
            return 0
        if pao2_fio2 < 400 and pao2_fio2 >= 300:
            return 1
        if pao2_fio2 < 300 and pao2_fio2 >= 200:
            return 2
    # for scores 3 and 4, respiratory support is required
        if pao2_fio2 < 200 and pao2_fio2 >= 100 and respiratory_support:
            return 3
        if pao2_fio2 < 100 and respiratory_support:
            return 4
    # if PF <200 but no respiratory support -> score 2 (per original SOFA)
        if pao2_fio2 < 200:
            return 2
    # fallback (should not reach)
        return 2
   
    def calculate_sofa_coagulation(self, platelets: float) -> int:
        """Platelets in ×10^3/µL (i.e. input 150 means 150)."""
        if platelets >= 150:
            return 0
        elif platelets >= 100:
            return 1
        elif platelets >= 50:
            return 2
        elif platelets >= 20:
            return 3
        else:
            return 4
    
    def calculate_sofa_liver(self, bilirubin: float) -> int:
        if bilirubin < 1.2:
            return 0
        elif bilirubin <= 1.9:
            return 1
        elif bilirubin <= 5.9:
            return 2
        elif bilirubin <= 11.9:
            return 3
        else:
            return 4
    
    def calculate_sofa_cardiovascular(self, map_mmhg: float, dopamine_dose: float = 0, 
                                     dobutamine_dose: float = 0, epinephrine_dose: float = 0, 
                                     norepinephrine_dose: float = 0) -> int:
        """
        Doses expected in mcg/kg/min. Return 0-4 per SOFA.
        """

        # 0 and 1 are based on MAP when no vasopressors used
        if all(d == 0 for d in [dopamine_dose, dobutamine_dose, epinephrine_dose, norepinephrine_dose]):
            if map_mmhg >= 70:
                return 0
            else:
                return 1

            # presence of vasopressors => score depends on dose
            # score 2: dopamine <= 5 OR dobutamine (any dose)
        if (dopamine_dose > 0 and dopamine_dose <= 5) or (dobutamine_dose > 0):
            return 2
            # score 3: dopamine 5.1-15 OR epinephrine <=0.1 OR norepinephrine <=0.1
        if (dopamine_dose > 5 and dopamine_dose <= 15) or \
            (epinephrine_dose > 0 and epinephrine_dose <= 0.1) or \
            (norepinephrine_dose > 0 and norepinephrine_dose <= 0.1):
            return 3
            # score 4: dopamine > 15 OR epinephrine > 0.1 OR norepinephrine > 0.1
        if dopamine_dose > 15 or epinephrine_dose > 0.1 or norepinephrine_dose > 0.1:
            return 4

            # fallback (should not reach)
        return 1
    
    def calculate_sofa_cns(self, gcs: int) -> int:
        if gcs == 15:
            return 0
        if 13 <= gcs <= 14:
            return 1
        elif 10 <= gcs <=12 :
            return 2
        elif 6 <= gcs <= 9:
            return 3
        else:
            return 4
    
    def calculate_sofa_renal(self, creatinine: float, urine_output_ml_day: Optional[float] = None) -> int:
        creat_score = 0
        if creatinine >= 5.0:
            creat_score = 4
        elif creatinine >= 3.5:
            creat_score = 3
        elif creatinine >= 2.0:
            creat_score = 2
        elif creatinine >= 1.2:
            creat_score = 1
        else:
            creat_score = 0

        if urine_output_ml_day is  None:
            return creat_score
        
        if urine_output_ml_day < 200:
            urine_score = 4
        elif urine_output_ml_day < 500:
            urine_score = 1
        else:
            urine_score = 0
        return max(creat_score, urine_score)
       
    
    def calculate_total_sofa(self, pao2_fio2: float, platelets: float, bilirubin: float,
                             map_mmhg: float, gcs: int, creatinine: float,
                             respiratory_support: bool = False, dopamine_dose: float = 0,
                             dobutamine_dose: float = 0, epinephrine_dose: float = 0,
                             norepinephrine_dose: float = 0, urine_output_ml_day: Optional[float] = None,
                             baseline_sofa: int = 0) -> Dict[str, Union[int, bool]]:
        scores = {
            'respiration': self.calculate_sofa_respiration(pao2_fio2, respiratory_support),
            'coagulation': self.calculate_sofa_coagulation(platelets),
            'liver': self.calculate_sofa_liver(bilirubin),
            'cardiovascular': self.calculate_sofa_cardiovascular(map_mmhg, dopamine_dose, 
                                                                dobutamine_dose, epinephrine_dose, 
                                                                norepinephrine_dose),
            'cns': self.calculate_sofa_cns(gcs),
            'renal': self.calculate_sofa_renal(creatinine, urine_output_ml_day)
        }
        total_sofa = sum(scores.values())
        delta_sofa = total_sofa - baseline_sofa
        sepsis_criteria_met = delta_sofa >= 2
        mortality_risk_approx = min(10 + (total_sofa * 5), 90)
        return {
            'individual_scores': scores,
            'total_sofa': total_sofa,
            'baseline_sofa': baseline_sofa,
            'delta_sofa': delta_sofa,
            'sepsis_criteria_met': sepsis_criteria_met,
            'estimated_mortality_risk_percent': mortality_risk_approx,
            'interpretation': self._interpret_sofa_score(total_sofa, sepsis_criteria_met)
        }
    
    def _interpret_sofa_score(self, total_sofa: int, sepsis_criteria: bool) -> str:
        if total_sofa == 0:
            return "Normal organ function - no signs of organ dysfunction"
        elif total_sofa <= 6:
            interpretation = "Mild to moderate organ dysfunction"
        elif total_sofa <= 12:
            interpretation = "Severe organ dysfunction"
        else:
            interpretation = "Very severe organ dysfunction - critical condition"
        if sepsis_criteria:
            interpretation += " - SEPSIS CRITERIA MET (acute increase ≥2 points)"
        return interpretation
    
    # ---------- qSOFA Methods ----------
    def calculate_qsofa(self, respiratory_rate: Optional[float], systolic_bp: Optional[float], gcs: Optional[int],
                        suspected_infection: bool = True) -> Dict[str, Union[int, bool, str]]:
        
        """Compute qSOFA score (0–3) using bedside criteria."""
        if any(v is None for v in [respiratory_rate, systolic_bp, gcs]):
            return {
                'qsofa_score': None,
                'error': 'Missing one or more required inputs (respiratory rate, SBP, or GCS).'
            }
        
        criteria = {
            'respiratory_rate_22_or_higher': respiratory_rate >= 22,
            'altered_mentation': gcs < 15,
            'systolic_bp_100_or_lower': systolic_bp <= 100
        }
        qsofa_score = sum(criteria.values())
        high_risk = qsofa_score >= 2
        interpretation = self._interpret_qsofa(qsofa_score, high_risk, suspected_infection)
        return {
            'criteria_met': criteria,
            'qsofa_score': qsofa_score,
            'high_risk_for_poor_outcomes': high_risk,
            'suspected_infection': suspected_infection,
            'interpretation': interpretation,
            'recommendations': self._get_qsofa_recommendations(qsofa_score, suspected_infection)
        }
    
    def _interpret_qsofa(self, score: int, high_risk: bool, suspected_infection: bool) -> str:
        if not suspected_infection:
            return "qSOFA should be used in patients with suspected infection"
        if score == 0:
            return "Low risk - no qSOFA criteria met"
        elif score == 1:
            return "Intermediate risk - one qSOFA criterion met, monitor closely"
        elif score >= 2:
            return "HIGH RISK - qSOFA ≥2 suggests increased risk of poor outcomes (ICU stay, death)"
        return "Invalid qSOFA score."

    def _get_qsofa_recommendations(self, score: int, suspected_infection: bool) -> str:
        if not suspected_infection:
            return "Assess for infection source first"
        if score >= 2:
            return "Consider ICU evaluation, escalate care, consider full SOFA scoring"
        elif score == 1:
            return "Increase monitoring frequency, reassess regularly"
        else:
            return "Continue routine care with infection monitoring"
    
# ---------- Septic Shock ----------
    def assess_septic_shock(
        self,
        map_mmhg: float,
        lactate_mmol_l: float,
        on_vasopressors: bool,
        adequate_volume_resus: bool,
        sepsis_present: bool
    ) -> Dict[str, Union[bool, str]]:
        """
        Assess septic shock based on Sepsis-3 definition
        """

        criteria = {
            "sepsis_present": sepsis_present,
            "persistent_hypotension_on_vasopressors": on_vasopressors and map_mmhg < 65,
            "lactate_greater_than_2": lactate_mmol_l > 2.0,
            "adequate_volume_resuscitation": adequate_volume_resus
        }

        septic_shock = all(criteria.values())

        return {
            "criteria": criteria,
            "septic_shock_present": septic_shock,
            "estimated_mortality_risk": ">40%" if septic_shock else "Variable",
            "interpretation": self._interpret_septic_shock(septic_shock, criteria)
        }


    def _interpret_septic_shock(
        self,
        shock_present: bool,
        criteria: Dict
    ) -> str:
        """
        Generate human-readable septic shock interpretation
        """

        if shock_present:
            return (
                "SEPTIC SHOCK PRESENT — Persistent hypotension requiring vasopressors "
                "with elevated lactate despite adequate fluid resuscitation. "
                "Hospital mortality >40%. Immediate ICU care required."
            )

        reasons = []

        if not criteria.get("sepsis_present"):
            reasons.append("Sepsis diagnosis not confirmed")

        if not criteria.get("persistent_hypotension_on_vasopressors"):
            reasons.append("No persistent hypotension despite vasopressors")

        if not criteria.get("lactate_greater_than_2"):
            reasons.append("Lactate ≤ 2 mmol/L")

        if not criteria.get("adequate_volume_resuscitation"):
            reasons.append("Adequate fluid resuscitation not documented")

        return "Septic shock criteria not fully met. Reasons: " + "; ".join(reasons)

    
    # ---------- NEWS2 Interpretation Helper (New) ----------
    def _interpret_news2(self, total_score: int, score_map: Dict) -> Tuple[str, str]:
        has_score_3 = 3 in score_map.values()
        
        if total_score >= 7:
            risk = "High"
            recommendation = "EMERGENCY RESPONSE: Clinical urgency high. Immediate senior clinical review and continuous monitoring required."
        elif total_score >= 5:
            risk = "Medium"
            recommendation = "URGENT REVIEW: Review by a clinician/doctor competent in acute illness. Monitor every hour."
        elif has_score_3: 
            risk = "Low-Medium"
            recommendation = "ROUTINE MONITORING:  Urgent Ward-based care. Monitor every 4-12 hours."
        else :
            risk = "Low"
            recommendation = "ROUTINE MONITORING:  Ward-based care. Monitor every 4-12 hours."
      

        return risk, recommendation

    # ---------- NEWS2 Calculation (Updated) ----------
    def calculate_news2(self, respiratory_rate: int, SpO2_Scale_1: int, SpO2_Scale_2: int,
                        supplemental_oxygen: bool, systolic_bp: int,
                        heart_rate: int, level_of_consciousness: str,
                        temperature: float,Age: int) -> Dict[str, Union[int, str]]:
        """
        Calculate NEWS2 score based on vital signs.
        level_of_consciousness expects: 'Alert', 'Voice', 'Pain', 'Unresponsive'
        """
        score = {}

        # Respiratory rate
        if 12 <= respiratory_rate <= 20:
            score['respiratory_rate'] = 0
        elif 9 <= respiratory_rate <= 11:
            score['respiratory_rate'] = 1
        elif 21 <= respiratory_rate <= 24:
            score['respiratory_rate'] = 2
        elif respiratory_rate <=8 or respiratory_rate >=32:
            score['respiratory_rate'] = 3

        # SpO2 Scale 1 (%)
        if SpO2_Scale_1 >= 96:
            score['oxygen_saturation'] = 0
        elif 94 <= SpO2_Scale_1 <= 95:
            score['SpO2_Scale_1'] = 1
        elif 92 <= SpO2_Scale_1 <= 93:
            score['SpO2_Scale_1'] = 2
        else:  # ≤91
            score['SpO2_Scale_1'] = 3

        # SpO2 Scale 2 (%)
        if 88 <= SpO2_Scale_2 >= 92 or (not supplemental_oxygen and SpO2_Scale_2 >=93)  :
            score['oxygen_saturation'] = 0
        elif 86 <= SpO2_Scale_2 <= 87 or (supplemental_oxygen and 93 <= SpO2_Scale_2 >= 94) :
            score['SpO2_Scale_2'] = 1
        elif 84 <= SpO2_Scale_2 <= 85 or (supplemental_oxygen and 95 <= SpO2_Scale_2 >= 96) :
            score['SpO2_Scale_2'] = 2
        else:  # ≤91
            score['SpO2_Scale_2'] = 3

        if Age<40:
            score['Age']=0
        elif 40<=Age<=65:
            score['Age']=1
        elif 66<=Age<=79:
            score['Age']=2
        elif Age>=80:
            score['Age']=3


        # Supplemental oxygen
        score['Fio2'] = 2 if supplemental_oxygen else 0

        # Systolic BP
        if 111 <= systolic_bp <= 219:
            score['systolic_bp'] = 0
        elif 101 <= systolic_bp <= 110:
            score['systolic_bp'] = 1
        elif 91 <= systolic_bp <= 100:
            score['systolic_bp'] = 2
        else:  # ≤90 or ≥220
            score['systolic_bp'] = 3

        # Heart rate
        if 51 <= heart_rate <= 90:
            score['heart_rate'] = 0
        elif 41 <= heart_rate <= 50 or 91 <= heart_rate <= 110:
            score['heart_rate'] = 1
        elif 111 <= heart_rate <= 130:
            score['heart_rate'] = 2
        else:  # ≤40 or ≥131
            score['heart_rate'] = 3

        # Level of consciousness (Updated to handle full word strings from form)
        # Assuming form sends 'Alert', 'Voice', 'Pain', or 'Unresponsive'
        if level_of_consciousness.upper().startswith('A'):
             score['level_of_consciousness'] = 0
        else: 
             score['level_of_consciousness'] = 3

        # Temperature
        if 36.1 <= temperature <= 38.0:
            score['temperature'] = 0
        elif 35.1 <= temperature <= 36.0 or 38.1 <= temperature <= 39.0:
            score['temperature'] = 1
        elif temperature <= 35.0:  # ≤35.0 or ≥39.1
            score['temperature'] = 3
        else: #>=39.1
            score['temperature'] = 2

        total_score = sum(score.values())

        # Use helper to get consistent risk and recommendation strings
        risk_level, interpretation = self._interpret_news2(total_score, score)

        # FIX: Return standardized keys required by app.py
        return {
            'individual_scores': score,
            'total_score': total_score,        # Standardized key
            'risk_band': risk_level,           # Standardized key
            'interpretation': interpretation     # Interpretation text
        }


# ---------- Example Usage ----------
def example_usage():
    scorer = SepsisScoring()
    
    print("=== SEPSIS SCORING SYSTEM CALCULATOR ===\n")
    
    # SOFA/qSOFA Example
    patient1_sofa = scorer.calculate_total_sofa(
        pao2_fio2=250, platelets=80, bilirubin=3.2,
        map_mmhg=65, gcs=12, creatinine=2.5,
        respiratory_support=True, dopamine_dose=8
    )
    patient1_qsofa = scorer.calculate_qsofa(
        respiratory_rate=28, systolic_bp=95, gcs=12
    )
    
    print("Patient 1 SOFA/qSOFA:")
    print(f"SOFA Score: {patient1_sofa['total_sofa']}, Delta: {patient1_sofa['delta_sofa']}, Sepsis Criteria Met: {patient1_sofa['sepsis_criteria_met']}")
    print(f"qSOFA Score: {patient1_qsofa['qsofa_score']}, High Risk: {patient1_qsofa['high_risk_for_poor_outcomes']}")
    print(f"SOFA Interpretation: {patient1_sofa['interpretation']}\n")
    
    # NEWS2 Example
    patient_news2 = scorer.calculate_news2(
        respiratory_rate=26, oxygen_saturation=90, supplemental_oxygen=True,
        systolic_bp=88, heart_rate=135, level_of_consciousness='Alert', temperature=39.2
    )
    
    print("Patient 1 NEWS2:")
    print(f"Total NEWS2 Score: {patient_news2['total_score']}")
    print(f"Risk Level: {patient_news2['risk_band']}")
    print(f"Interpretation: {patient_news2['interpretation']}")
    print(f"Individual Scores: {patient_news2['individual_scores']}\n")
    
    # Septic Shock Example
    shock_assessment = scorer.assess_septic_shock(
        map_mmhg=68, lactate_mmol_l=3.5,
        on_vasopressors=True, adequate_volume_resus=True, sepsis_present=True
    )
    
    print("Septic Shock Assessment:")
    print(f"Septic Shock Present: {shock_assessment['septic_shock_present']}")
    print(f"Estimated Mortality Risk: {shock_assessment['estimated_mortality_risk']}")
    print(f"Interpretation: {shock_assessment['interpretation']}\n")


if __name__ == "__main__":
    example_usage()
