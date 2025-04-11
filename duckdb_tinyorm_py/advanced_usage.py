"""
Example of advanced usage for duckdb-tinyorm-py
"""
import asyncio
import pandas as pd
from .repository import (
    DuckDbRepository, BaseRepository, QueryBuilder,
    DuckDbLocation, DuckDbConfig
)
from .decorators import (
    entity, id_field, field, index, repository
)


# Create a more advanced DuckDB config
config = DuckDbConfig(
    name='advanced',
    location=DuckDbLocation.FILE,
    filename='my_database.db',
    settings={
        'memory_limit': "'1GB'", # Keep quotes for string values like memory limits
        # 'threads': 'auto'      # Remove extra quotes for keywords like 'auto'
    },
    load_extensions=True,
    extensions=['json']
)

# Create DuckDB connection
duck_db = DuckDbRepository.get_instance(config)


# Define an entity with more advanced features
@entity(table_name="courses")
@index(["name"], unique=True)
@index(["department", "year"])
class Course:
    def __init__(self, id_=None, name="", description="", department="", year=None, 
                credits=0, active=True, metadata=None):
        # Assign to the backing attribute _id
        self._id = id_ 
        # Use private attributes for storage if using properties
        self._name = name
        self._description = description
        self._department = department
        self._year = year or 2025
        self._credits = credits
        self._active = active
        self._metadata = metadata or {}
    
    @property # Add @property decorator
    @id_field('INTEGER', auto_increment=True)
    def id(self) -> int:
        # Return the backing attribute _id
        return self._id
    
    @id.setter
    def id(self, value: int):
        # Set the backing attribute _id
        self._id = value

    @property  # Use @property for the getter
    @field('VARCHAR', not_null=True, unique=True) # Apply @field to the property getter
    def name(self) -> str:
        return self._name

    @name.setter # Standard setter syntax
    def name(self, value: str):
        self._name = value
    
    @property # Use @property
    @field('TEXT')
    def description(self) -> str:
        return self._description

    @description.setter # Standard setter
    def description(self, value: str):
        self._description = value
    
    @property # Use @property
    @field('VARCHAR', not_null=True)
    def department(self) -> str:
        return self._department

    @department.setter # Standard setter
    def department(self, value: str):
        self._department = value
    
    @property # Use @property
    @field('INTEGER', not_null=True)
    def year(self) -> int:
        return self._year

    @year.setter # Standard setter
    def year(self, value: int):
        self._year = value
    
    @property # Use @property
    @field('INTEGER', default=3)
    def credits(self) -> int:
        return self._credits

    @credits.setter # Standard setter
    def credits(self, value: int):
        # Consider adding validation from validate() here if appropriate
        self._credits = value
    
    @property # Use @property
    @field('BOOLEAN', default=True)
    def active(self) -> bool:
        return self._active

    @active.setter # Standard setter
    def active(self, value: bool):
        self._active = value
    
    @property # Use @property
    @field('JSON')
    def metadata(self) -> dict:
        return self._metadata

    @metadata.setter # Standard setter
    def metadata(self, value: dict):
        self._metadata = value
    
    def validate(self):
        """Custom validation logic"""
        # Access attributes via self.<name> (which uses the property getters)
        if self.credits < 0 or self.credits > 6:
            raise ValueError("Credits must be between 0 and 6")
        
        if not self.name:
            raise ValueError("Course name cannot be empty")


# Create a repository for the entity
@repository(Course)
class CourseRepository(BaseRepository[Course, int]):
    def __init__(self, db=None):
        super().__init__(db or duck_db)
    
    async def find_active_courses(self):
        """Example of custom repository method"""
        return await self.find_by({"active": True})
    
    async def find_by_department(self, department):
        """Find courses by department"""
        return await self.find_by({"department": department})
    
    async def find_by_credits(self, min_credits, max_credits):
        """Find courses with credits in range"""
        query = self.query().where("credits", ">=", min_credits).and_where("credits", "<=", max_credits)
        return await self.execute_query(query)
    
    async def deactivate_all_in_department(self, department):
        """Deactivate all courses in a department"""
        courses = await self.find_by_department(department)
        
        for course in courses:
            course.active = False
            await self.save(course)
        
        return len(courses)
    
    async def export_to_dataframe_with_filters(self, department=None, active_only=True):
        """Export courses to DataFrame with filters"""
        query = self.query()
        
        if department:
            query.where("department", department)
        
        if active_only:
            query.and_where("active", True)
        
        return self.to_dataframe(query)
    
    async def export_to_parquet(self, filename):
        """Export all courses to a Parquet file"""
        try:
            await self.to_parquet(path=filename)
            return f"Exported to {filename}"
        except ImportError:
            return "Export failed: Please install pyarrow or fastparquet package for Parquet support"


async def main():
    # Initialize the repository
    course_repo = CourseRepository()
    await course_repo.init(drop_if_exists=True)
    
    # Create and save entities
    courses = [
        Course(name="Python Programming", description="Learn Python basics", department="CS", year=2025, credits=3),
        Course(name="Data Science", description="Introduction to data science", department="CS", year=2025, credits=4),
        Course(name="Machine Learning", description="ML algorithms and applications", department="CS", year=2025, credits=4),
        Course(name="Database Systems", description="Database design and SQL", department="CS", year=2025, credits=3),
        Course(name="Statistics", description="Introduction to statistics", department="MATH", year=2025, credits=3, 
              metadata={"prerequisites": ["Calculus I", "Calculus II"]}),
    ]
    
    # Save all courses in one transaction
    saved_courses = await course_repo.save_all(courses)
    
    print(f"Saved {len(saved_courses)} courses")
    
    # Demonstrate various query methods
    print("\n--- Finding all courses ---")
    all_courses = await course_repo.find_all()
    for course in all_courses:
        print(f"{course.id}: {course.name} ({course.department}, {course.credits} credits)")
    
    print("\n--- Finding CS department courses ---")
    cs_courses = await course_repo.find_by_department("CS")
    for course in cs_courses:
        print(f"{course.id}: {course.name} ({course.credits} credits)")
    
    print("\n--- Finding courses with 4 credits ---")
    courses_4_credits = await course_repo.find_by_credits(4, 4)
    for course in courses_4_credits:
        print(f"{course.id}: {course.name} ({course.department})")
    
    # Use advanced query builder
    print("\n--- Complex query example ---")
    query = course_repo.query() \
        .select("id", "name", "credits") \
        .where("active", True) \
        .and_where("department", "CS") \
        .order_by("credits", "DESC") \
        .limit(2)
    
    top_cs_courses = await course_repo.execute_query(query)
    for course in top_cs_courses:
        print(f"{course.id}: {course.name} ({course.credits} credits)")
    
    # Export to DataFrame
    print("\n--- Exporting to DataFrame ---")
    df = await course_repo.export_to_dataframe_with_filters(department="CS")
    print(df.head())
    
    # Export to Parquet
    print("\n--- Exporting to Parquet ---")
    result = await course_repo.export_to_parquet("courses.parquet")
    print(result)
    
    # Count entities
    cs_count = await course_repo.count({"department": "CS"})
    print(f"\nTotal CS courses: {cs_count}")
    
    # Demonstrate transaction safety
    print("\n--- Transaction example ---")
    try:
        # Start transaction explicitly
        duck_db.begin_transaction()
        
        # This will succeed
        new_course = Course(name="Web Development", description="HTML, CSS, JS", 
                           department="CS", year=2025, credits=3)
        await course_repo.save(new_course)
        
        # This will fail validation and trigger rollback
        invalid_course = Course(name="Invalid Course", department="CS", year=2025, credits=10)
        await course_repo.save(invalid_course)
        
        duck_db.commit()
    except Exception as e:
        duck_db.rollback()
        print(f"Transaction rolled back: {str(e)}")
    
    # Verify rollback
    all_courses = await course_repo.find_all()
    print(f"\nTotal courses after transaction: {len(all_courses)}")
    
    # Export to JSON
    json_data = await course_repo.to_json()
    print("\n--- JSON Export Sample ---")
    print(json_data[:200] + "...")  # Show first 200 chars


if __name__ == "__main__":
    asyncio.run(main())