from api import db, app
from api.models import Report, Puf, Cancer
import project


# Remove spontaneous quoting of column name
db.engine.dialect.identifier_preparer.initial_quote = ''
db.engine.dialect.identifier_preparer.final_quote = ''

#Create database and tables
db.create_all()
print "Table(s) schema created, inserting data..."

# Insert Data -- Bulk insert of DataFrame
df_report = project.readCSV()
report_lst = df_report.to_dict(orient='records')  # orient by records to align format
db.session.execute(Report.__table__.insert(), report_lst)
db.session.commit()
df_puf = project.readPUF()
for elem in df_puf[:5]:
    puf_lst = elem.to_dict(orient='records')
    db.session.execute(Puf.__table__.insert(), puf_lst)
    db.session.commit()
df_can = project.readBCH()
can_lst = df_can.to_dict(orient='records')
db.session.execute(Cancer.__table__.insert(), can_lst)

#Commit changes
db.session.commit()
db.session.close()  # close session
print "Data insert successful...database initialization complete"
