from duckdb_tinyorm_py import entity, field, id_field,BaseRepository, repository, DuckDbConfig, DuckDbLocation
import asyncio

@entity(table_name="courses")
class Course:
    def __init__(self, id_=None, name="", department=""):
        self._id = id_
        self._name = name
        self._department = department

    @property
    @id_field('INTEGER', auto_increment=True)
    def id(self) -> int:
        return self._id
    
    @id.setter
    def id(self, value: int):
        self._id = value

    @property
    @field('VARCHAR', not_null=True)
    def name(self) -> str:
        return self._name
        
    @name.setter
    def name(self, value: str):
        self._name = value
        
    @property
    @field('VARCHAR', not_null=True)
    def department(self) -> str:
        return self._department
    
    @department.setter
    def department(self, value: str):
        self._department = value



# Configure your database
config = DuckDbConfig(
    name='mydb',
    location=DuckDbLocation.FILE,
    filename='my_database.db'
)

@repository(Course)
class CourseRepository(BaseRepository[Course, int]):
    # Custom repository methods
    async def find_by_department(self, department):
        """Find courses by department"""
        return await self.find_by({"department": department})
    


async def main():
    # Initialize repository and create table
    repo = CourseRepository()
    await repo.init()
    
    # Create and save entities
    course = Course(name="Python Programming", department="CS")
    saved = await repo.save(course)
    print(f"Saved course ID: {saved.id}")
    
    # Find entities
    all_courses = await repo.find_all()
    cs_courses = await repo.find_by_department("CS")
    
    # Update an entity
    course.name = "Advanced Python"
    updated = await repo.save(course)
    
    # Delete an entity
    await repo.remove(course)
    
    # Export to different formats
    df = repo.to_dataframe()
    json_data = await repo.to_json()
    await repo.to_parquet(path="courses.parquet")

# Run the async code
asyncio.run(main())