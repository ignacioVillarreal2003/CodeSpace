export interface Category {
  id: string,
  title: string,
  color: string,
  image: string
}

export const categories: Category[] = [
  {
    id: "cat-001",
    title: "IA and Machine Learning",
    color: "#b6f4bc",
    image: "artificial_intelligence.png"
  }
];
