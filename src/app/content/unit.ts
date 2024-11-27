export interface Unit {
  id: string,
  topicId: string,
  title: string,
  color: string,
  content: string
}

export const units: Unit[] = [
  {
    id: 'unit-001',
    topicId: "top-001",
    title: 'Introducción a TensorFlow',
    color: "#C2FFC7",
    content: 'IA_and_Machine_Learning_for_Coders/Introduction to TensorFlow.md'
  },
  {
    id: 'unit-002',
    topicId: "top-001",
    title: 'Introducción a la visión artificial',
    color: "#C2FFC7",
    content: 'IA_and_Machine_Learning_for_Coders/Introduction to Computer Vision.md'
  },
  {
    id: 'unit-003',
    topicId: "top-001",
    title: 'Más Allá de lo Básico: Detectando Características en Imágenes',
    color: "#C2FFC7",
    content: 'IA_and_Machine_Learning_for_Coders/Going_Beyond_the_Basics_Detecting_Features_in_Images.md'
  },
];
