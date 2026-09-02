/**
 * Metadata for every field in the prediction form.
 *
 * This MUST match backend/app/schemas/prediction.py field-for-field (same
 * names, same value ranges) — that Pydantic schema is the real source of
 * truth for what's valid; this file just describes how to render each
 * field as a form control with a human-readable label (brief 4.4: "No raw
 * feature names ... without a human-readable label in the main UI").
 */

const yesNo = [
  { value: 1, label: "Yes" },
  { value: 0, label: "No" },
];

export const FORM_SECTIONS = [
  {
    title: "Application",
    fields: [
      { key: "marital_status", label: "Marital Status", type: "number", min: 1, max: 6, help: "UCI code 1–6 (1 = Single)" },
      { key: "application_mode", label: "Application Mode", type: "number", min: 1, help: "UCI application mode code" },
      { key: "application_order", label: "Application Preference Order", type: "number", min: 0, help: "0 = first choice course" },
      { key: "course", label: "Course", type: "number", min: 1, help: "UCI course code" },
      { key: "nationality", label: "Nationality", type: "number", min: 1, help: "UCI nationality code (1 = Portuguese)" },
      { key: "previous_qualification", label: "Previous Qualification Type", type: "number", min: 1, help: "UCI qualification type code" },
      { key: "previous_qualification_grade", label: "Previous Qualification Grade", type: "number", min: 0, max: 200, step: 0.1 },
      { key: "admission_grade", label: "Admission Grade", type: "number", min: 0, max: 200, step: 0.1 },
      { key: "age_at_enrollment", label: "Age at Enrollment", type: "number", min: 15, max: 100 },
      { key: "daytime_evening_attendance", label: "Daytime Attendance", type: "select", options: yesNo },
      { key: "displaced", label: "Displaced Student", type: "select", options: yesNo },
      { key: "international", label: "International Student", type: "select", options: yesNo },
      { key: "gender", label: "Gender", type: "select", options: [{ value: 0, label: "Female" }, { value: 1, label: "Male" }] },
    ],
  },
  {
    title: "Family Background",
    fields: [
      { key: "mothers_qualification", label: "Mother's Education Level", type: "number", min: 1, help: "UCI qualification code" },
      { key: "fathers_qualification", label: "Father's Education Level", type: "number", min: 1, help: "UCI qualification code" },
      { key: "mothers_occupation", label: "Mother's Occupation", type: "number", min: 1, help: "UCI occupation code" },
      { key: "fathers_occupation", label: "Father's Occupation", type: "number", min: 1, help: "UCI occupation code" },
    ],
  },
  {
    title: "Financial & Support Status",
    fields: [
      { key: "debtor", label: "Has Outstanding Debt", type: "select", options: yesNo },
      { key: "tuition_fees_up_to_date", label: "Tuition Fees Up To Date", type: "select", options: yesNo },
      { key: "scholarship_holder", label: "Scholarship Holder", type: "select", options: yesNo },
      { key: "educational_special_needs", label: "Educational Special Needs", type: "select", options: yesNo },
    ],
  },
  {
    title: "1st Semester Academic Record",
    fields: [
      { key: "curricular_units_1st_sem_credited", label: "Units Credited", type: "number", min: 0 },
      { key: "curricular_units_1st_sem_enrolled", label: "Units Enrolled", type: "number", min: 0 },
      { key: "curricular_units_1st_sem_evaluations", label: "Units Evaluated", type: "number", min: 0 },
      { key: "curricular_units_1st_sem_approved", label: "Units Approved", type: "number", min: 0 },
      { key: "curricular_units_1st_sem_grade", label: "Average Grade", type: "number", min: 0, max: 20, step: 0.1 },
      { key: "curricular_units_1st_sem_without_evaluations", label: "Units Without Evaluation", type: "number", min: 0 },
    ],
  },
  {
    title: "2nd Semester Academic Record",
    fields: [
      { key: "curricular_units_2nd_sem_credited", label: "Units Credited", type: "number", min: 0 },
      { key: "curricular_units_2nd_sem_enrolled", label: "Units Enrolled", type: "number", min: 0 },
      { key: "curricular_units_2nd_sem_evaluations", label: "Units Evaluated", type: "number", min: 0 },
      { key: "curricular_units_2nd_sem_approved", label: "Units Approved", type: "number", min: 0 },
      { key: "curricular_units_2nd_sem_grade", label: "Average Grade", type: "number", min: 0, max: 20, step: 0.1 },
      { key: "curricular_units_2nd_sem_without_evaluations", label: "Units Without Evaluation", type: "number", min: 0 },
    ],
  },
  {
    title: "Regional Economic Context",
    fields: [
      { key: "unemployment_rate", label: "Unemployment Rate (%)", type: "number", step: 0.1 },
      { key: "inflation_rate", label: "Inflation Rate (%)", type: "number", step: 0.1 },
      { key: "gdp", label: "GDP Growth", type: "number", step: 0.01 },
    ],
  },
];

// A realistic, plausible default so the form isn't empty/intimidating on
// first load — an advisor can tweak from here rather than starting blank.
export const DEFAULT_FORM_VALUES = {
  marital_status: 1,
  application_mode: 17,
  application_order: 1,
  course: 9254,
  nationality: 1,
  mothers_qualification: 1,
  fathers_qualification: 1,
  mothers_occupation: 1,
  fathers_occupation: 1,
  previous_qualification: 1,
  daytime_evening_attendance: 1,
  displaced: 0,
  educational_special_needs: 0,
  debtor: 0,
  tuition_fees_up_to_date: 1,
  gender: 0,
  scholarship_holder: 0,
  international: 0,
  previous_qualification_grade: 126.0,
  admission_grade: 122.6,
  age_at_enrollment: 20,
  curricular_units_1st_sem_credited: 0,
  curricular_units_1st_sem_enrolled: 6,
  curricular_units_1st_sem_evaluations: 6,
  curricular_units_1st_sem_approved: 5,
  curricular_units_1st_sem_grade: 12.4,
  curricular_units_1st_sem_without_evaluations: 0,
  curricular_units_2nd_sem_credited: 0,
  curricular_units_2nd_sem_enrolled: 6,
  curricular_units_2nd_sem_evaluations: 6,
  curricular_units_2nd_sem_approved: 4,
  curricular_units_2nd_sem_grade: 11.8,
  curricular_units_2nd_sem_without_evaluations: 0,
  unemployment_rate: 10.8,
  inflation_rate: 1.4,
  gdp: 1.74,
};

// Human-readable labels for feature keys as returned by /explain's
// top_factors (already human-readable from the backend, but this covers
// any other place a raw key might need display, e.g. student feature dumps).
export const FEATURE_LABEL_BY_KEY = FORM_SECTIONS
  .flatMap((s) => s.fields)
  .reduce((acc, f) => ({ ...acc, [f.key]: f.label }), {});
